#include "model_quintom_smg.h"
#include "parallel.h"


/**
 * Deep-copy a fzerofun_workspace into dst.
 * dst must be freed with fzerofun_workspace_free_clone() when done.
 *
 * @param src                    workspace to clone
 * @param unknown_parameters_size number of entries in unknown_parameters_index
 * @param dst                    output: populated clone
 * @param errmsg                 error message buffer
 */
static int fzerofun_workspace_clone(
    const struct fzerofun_workspace *src,
    int unknown_parameters_size,
    struct fzerofun_workspace *dst,
    ErrorMsg errmsg)
{
  /* Deep-copy fc (allocates name/value/read arrays and copies contents) */
  class_call(parser_init_from_pfc((struct file_content *)&src->fc, &dst->fc, errmsg), errmsg, errmsg);

  dst->target_size = src->target_size;
  dst->required_computation_stage = src->required_computation_stage;

  dst->unknown_parameters_index = (int *)malloc(unknown_parameters_size * sizeof(int));
  class_test(dst->unknown_parameters_index == NULL, errmsg, "Could not allocate unknown_parameters_index clone");
  memcpy(dst->unknown_parameters_index, src->unknown_parameters_index,
         unknown_parameters_size * sizeof(int));

  dst->target_name = (enum target_names *)malloc(src->target_size * sizeof(enum target_names));
  class_test(dst->target_name == NULL, errmsg, "Could not allocate target_name clone");
  memcpy(dst->target_name, src->target_name,
         src->target_size * sizeof(enum target_names));

  dst->target_value = (double *)malloc(src->target_size * sizeof(double));
  class_test(dst->target_value == NULL, errmsg, "Could not allocate target_value clone");
  memcpy(dst->target_value, src->target_value,
         src->target_size * sizeof(double));

  return _SUCCESS_;
}

/**
 * Free heap members allocated by fzerofun_workspace_clone().
 * Does NOT free the dst struct itself (caller owns it).
 */
static void fzerofun_workspace_free_clone(struct fzerofun_workspace *dst)
{
  parser_free(&dst->fc);
  free(dst->unknown_parameters_index);
  free(dst->target_name);
  free(dst->target_value);
}


/**
 * Scan the boundary of [phi_low,phi_high] x [alpha_low,alpha_high], compute the winding
 * number of the residual vector field, and track the boundary point closest to zero.
 *
 * @param phi_low,phi_high   phi interval bounds
 * @param alpha_low,alpha_high alpha interval bounds
 * @param N_phi,N_alpha      lattice sizes along each axis (must be >= 2)
 * @param pfzw               fzerofun workspace
 * @param fevals             running function-evaluation counter (incremented in place)
 * @param root_loc           in/out: (phi,alpha) of the current best guess (updated if boundary improves it)
 * @param root_measure       in/out: |F|^2 of the current best guess (updated in place)
 * @param winding_number output: |winding number| / (2*pi)
 * @param errmsg             error message buffer
 */
static int compute_winding_number_2d(
    struct fzerofun_workspace *pfzw,
    double phi_low, double phi_high,
    double alpha_low, double alpha_high,
    int N_phi, int N_alpha,
    int *fevals,
    double root_loc[2], double *root_measure,
    double *winding_number,
    ErrorMsg errmsg)
{
  int winding_N = 2 * (N_alpha + N_phi) - 4;
  double step_phi   = (phi_high   - phi_low)   / (N_phi   - 1);
  double step_alpha = (alpha_high - alpha_low) / (N_alpha - 1);

  /* -- 1. Pre-compute all boundary coordinates in winding order -- */
  double (*boundary_xt)[2] = (double (*)[2])malloc(winding_N * sizeof(*boundary_xt));
  class_test(boundary_xt == NULL, errmsg, "Could not allocate boundary_xt");
  double (*boundary_F)[2]  = (double (*)[2])malloc(winding_N * sizeof(*boundary_F));
  class_test(boundary_F  == NULL, errmsg, "Could not allocate boundary_F");

  int idx = 0;
  /* Bottom edge: alpha_low, phi low -> high */
  for (int i = 0; i < N_phi; i++){
    boundary_xt[idx][0] = phi_low + i * step_phi;
    boundary_xt[idx][1] = alpha_low;
    idx++;
  }
  /* Right edge: phi_high, alpha low -> high */
  for (int i = 1; i < N_alpha; i++){
    boundary_xt[idx][0] = phi_high;
    boundary_xt[idx][1] = alpha_low + i * step_alpha;
    idx++;
  }
  /* Top edge: alpha_high, phi high -> low */
  for (int i = 1; i < N_phi; i++){
    boundary_xt[idx][0] = phi_high - i * step_phi;
    boundary_xt[idx][1] = alpha_high;
    idx++;
  }
  /* Left edge: phi_low, alpha high -> low (skipping corners already covered) */
  for (int i = 1; i < N_alpha - 1; i++){
    boundary_xt[idx][0] = phi_low;
    boundary_xt[idx][1] = alpha_high - i * step_alpha;
    idx++;
  }

  /* -- 2. Pre-clone one workspace per boundary point -- */
  struct fzerofun_workspace *clones = (struct fzerofun_workspace *)malloc(winding_N * sizeof(struct fzerofun_workspace));
  class_test(clones == NULL, errmsg, "Could not allocate workspace clones");
  for (int k = 0; k < winding_N; k++){
    class_call(fzerofun_workspace_clone(pfzw, 2, &clones[k], errmsg), errmsg, errmsg);
  }

  /* -- 3. Evaluate all boundary points in parallel -- */
  class_setup_parallel();

  for (int k = 0; k < winding_N; k++){
    class_run_parallel(with_arguments(k, boundary_xt, boundary_F, clones, errmsg),
      double xt[2];
      xt[0] = boundary_xt[k][0];
      xt[1] = boundary_xt[k][1];
      class_call(input_try_unknown_parameters(xt, 2, &clones[k], boundary_F[k], errmsg),
                 errmsg, errmsg);
      return _SUCCESS_;
    );
  }

  class_finish_parallel();

  /* -- 4. Free clones -- */
  for (int k = 0; k < winding_N; k++){
    fzerofun_workspace_free_clone(&clones[k]);
  }
  free(clones);

  *fevals += winding_N;

  /* -- 5. Min reduction: find boundary point closest to root -- */
  for (int k = 0; k < winding_N; k++){
    double rm = pow(boundary_F[k][0], 2.) + pow(boundary_F[k][1], 2.);
    if (rm < *root_measure){
      *root_measure    = rm;
      root_loc[0] = boundary_xt[k][0];
      root_loc[1] = boundary_xt[k][1];
    }
  }

  /* -- 6. Compute winding number from ordered boundary samples -- */
  double winding_number_integral = 0.;
  for (int i = 0; i < winding_N; i++){
    int ip = (i + 1) % winding_N;
    double cross = boundary_F[i][0] * boundary_F[ip][1] - boundary_F[i][1] * boundary_F[ip][0];
    double dot   = boundary_F[i][0] * boundary_F[ip][0] + boundary_F[i][1] * boundary_F[ip][1];
    winding_number_integral += atan2(cross, dot);
  }
  *winding_number = fabs(winding_number_integral) / (2. * _PI_);

  free(boundary_F);
  free(boundary_xt);
  return _SUCCESS_;
}

/**
 * Coarse-grain interior scan: evaluate F on a (N_phi x N_alpha) interior grid
 * (skipping the boundary) and update root_loc/root_measure if a closer-to-zero
 * point is found.
 */
static int find_coarse_root_2d(
    struct fzerofun_workspace *pfzw,
    double phi_low, double phi_high,
    double alpha_low, double alpha_high,
    int N_phi, int N_alpha,
    int *fevals,
    double root_loc[2], double *root_measure,
    ErrorMsg errmsg){

  double step_phi   = (phi_high   - phi_low)   / (N_phi   - 1);
  double step_alpha = (alpha_high - alpha_low) / (N_alpha - 1);
  int N_inner_phi   = N_phi   - 2;
  int N_inner_alpha = N_alpha - 2;
  int N_inner       = N_inner_phi * N_inner_alpha;

  /* -- 1. Pre-compute all interior grid coordinates (row-major, skipping boundary) -- */
  double (*grid_xt)[2] = (double (*)[2])malloc(N_inner * sizeof(*grid_xt));
  class_test(grid_xt == NULL, errmsg, "Could not allocate grid_xt");
  double (*grid_F)[2]  = (double (*)[2])malloc(N_inner * sizeof(*grid_F));
  class_test(grid_F  == NULL, errmsg, "Could not allocate grid_F");

  int k = 0;
  for (int i = 1; i < N_phi - 1; i++){
    for (int j = 1; j < N_alpha - 1; j++){
      grid_xt[k][0] = phi_low   + step_phi   * i;
      grid_xt[k][1] = alpha_low + step_alpha * j;
      k++;
    }
  }

  /* -- 2. Pre-clone one workspace per grid point -- */
  struct fzerofun_workspace *clones = (struct fzerofun_workspace *)malloc(N_inner * sizeof(struct fzerofun_workspace));
  class_test(clones == NULL, errmsg, "Could not allocate workspace clones");
  for (k = 0; k < N_inner; k++){
    class_call(fzerofun_workspace_clone(pfzw, 2, &clones[k], errmsg), errmsg, errmsg);
  }

  /* -- 3. Evaluate all interior points in parallel -- */
  class_setup_parallel();

  for (k = 0; k < N_inner; k++){
    class_run_parallel(with_arguments(k, grid_xt, grid_F, clones, errmsg),
      double xt[2];
      xt[0] = grid_xt[k][0];
      xt[1] = grid_xt[k][1];
      class_call(input_try_unknown_parameters(xt, 2, &clones[k], grid_F[k], errmsg),
                 errmsg, errmsg);
      return _SUCCESS_;
    );
  }

  class_finish_parallel();

  /* -- 4. Free clones -- */
  for (k = 0; k < N_inner; k++){
    fzerofun_workspace_free_clone(&clones[k]);
  }
  free(clones);

  *fevals += N_inner;

  /* -- 5. Min reduction: find interior point closest to root -- */
  for (k = 0; k < N_inner; k++){
    double rm = pow(grid_F[k][0], 2.) + pow(grid_F[k][1], 2.);
    if (rm < *root_measure){
      *root_measure = rm;
      root_loc[0]   = grid_xt[k][0];
      root_loc[1]   = grid_xt[k][1];
    }
  }

  free(grid_F);
  free(grid_xt);
  return _SUCCESS_;
}

/**
 * Related to 'shooting': Find the root of a one-dimensional
 * function. This function starts from a first guess, then uses a few
 * steps to bracket the root, and then calls another function to
 * actually get the root.
 *
 * @param xzero     Output: root x such that f(x)=0 up to tolerance (f(x) = input_fzerofun_1d)
 * @param fevals    Output: number of iterations (that is, of CLASS runs) needed to find the root
 * @param tol_x_rel Input : Relative tolerance compared to bracket of root that is used to find root.
 * @param pfzw      Input : pointer to workspace containing targets, unkown parameters and other relevant information
 * @param errmsg    Input/Output: Error message
 * @return the error status
 */

int input_find_root_quintom(double *xzeros,
                    int *fevals,
                    int unknown_parameters_size,
                    double tol_x_rel,
                    double tol_F,
                    struct fzerofun_workspace *pfzw,
                    int input_verbose,
                    ErrorMsg errmsg){

  /** Summary: */

  /** Define local variables */
  // Currently works only if x < 0 -> alpha = (1, oo)
  double phi_low, phi_high;
  
  int flag;
  double xi_smg, v0_smg, m2_smg, lambda_smg;
  // != Martin
  class_call(parser_read_double(&(pfzw->fc),"xi_smg",&xi_smg,&flag,errmsg),
            errmsg,
            errmsg);
  class_call(parser_read_double(&(pfzw->fc),"V0_smg",&v0_smg,&flag,errmsg),
            errmsg,
            errmsg);
  class_call(parser_read_double(&(pfzw->fc),"m2_smg",&m2_smg,&flag,errmsg),
            errmsg,
            errmsg);
  class_call(parser_read_double(&(pfzw->fc),"lambda_smg",&lambda_smg,&flag,errmsg),
            errmsg,
            errmsg);

  /*
    1) d_dphi(V/F^2)|_phi* <= 0 -> Otherwise it would be tilted towards negative side
    We want it to roll down from 0 -> phi+
    This gives that phi_*_max = sqrt(-m2/lambda)
    2) Check that V/F^2 at singular point limit (F(phi_0) = 0) is > -oo
      More strict is saying that V(phi_0) > 0
      */
  
  double V_determinant = m2_smg*m2_smg - 4.*lambda_smg*v0_smg;
  class_test(lambda_smg < 0, errmsg, "lambda (%g) < 0 is not valid", lambda_smg);
  class_test(-m2_smg/lambda_smg < 0, errmsg, "m^2/lambda < 0 -> Solution does not exist with these parameters. params (xi, v0, m2, lambda): (%g %g %g %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
  phi_low = 0.;
  phi_high = sqrt(-m2_smg/lambda_smg);

  /*
   * Determine valid phi intervals based on the sign of the V discriminant.
   *
   * V(phi) = v0 + m2/2*phi^2 + lambda/4*phi^4  has zeros at
   *   phi^2 = (-m2 ± sqrt(m2^2 - 4*lambda*v0)) / lambda
   *
   * det < 0:  V has no real zeros -> valid range is the full [0, phi_high]
   *
   * det >= 0: V has zeros at phi_star_1 <= phi_star_2, giving the excluded zone.
   *           Valid ranges are [0, phi_star_1] and [phi_star_2, phi_high],
   *           each clipped to [0, phi_high].
   *
   *    OK         Not OK        OK
   * --------|----------------|--------|
   *     phi_star_1      phi_star_2    phi_high
   */
  double phi_intervals[2][2];

  if (V_determinant < 0) {
    /* No real zeros of V: the full interval is valid */
    phi_intervals[0][0] = phi_low;
    phi_intervals[0][1] = phi_high;
    phi_intervals[1][0] = phi_high;  /* empty sentinel */
    phi_intervals[1][1] = phi_high;
    if (input_verbose >= 2) {
      printf("V determinant < 0: no V zeros, valid range [%g, %g]\n", phi_low, phi_high);
    }
  }
  else {
    /* Positive determinant: compute excluded-zone boundaries */
    double sqrt_det = sqrt(V_determinant);
    double phi_star_1 = sqrt((-m2_smg - sqrt_det) / lambda_smg) - pow(fabs(xi_smg), -0.5);
    double phi_star_2 = sqrt((-m2_smg + sqrt_det) / lambda_smg) - pow(fabs(xi_smg), -0.5);

    if (input_verbose >= 2) {
      printf("V determinant >= 0: phi_star_1 = %g, phi_star_2 = %g, phi_high = %g\n",
             phi_star_1, phi_star_2, phi_high);
      printf("Candidate ranges: [%g, %g] and [%g, %g] (before clipping)\n",
             phi_low, phi_star_1, phi_star_2, phi_high);
    }

    /* Interval 0: [phi_low, phi_star_1], clipped to phi_high */
    phi_intervals[0][0] = phi_low;
    phi_intervals[0][1] = (phi_star_1 < phi_high) ? phi_star_1 : phi_high;

    /* Interval 1: [phi_star_2, phi_high], clipped to phi_low */
    phi_intervals[1][0] = (phi_star_2 > phi_low) ? phi_star_2 : phi_low;
    phi_intervals[1][1] = phi_high;
  }
  
  /** 
   * =============================
   * ====== Tune only shift ======
   * =============================
   */
  if (unknown_parameters_size == 1){
    // NB! Assume this is scalar field then!
    // Valid search intervals: solution must be in (phi_low, phi_high) but not in (phi2_1, phi2_2)
    int N_tries = 10;
    double f1_phi, f2_phi;
    double dphi;
    double phi = phi_low;
    int found = 0;

    for (int iv = 0; iv < 2 && !found; iv++){
      double iv_low  = phi_intervals[iv][0];
      double iv_high = phi_intervals[iv][1];
      if (iv_high <= iv_low) continue;  // interval is empty/invalid

      class_call(input_fzerofun_1d(iv_low, pfzw, &f1_phi, errmsg), errmsg, errmsg);
      dphi = (iv_high - iv_low) / N_tries;
      for (int i = 1; i <= N_tries; i++){
        phi = iv_low + i * dphi;
        class_call(input_fzerofun_1d(phi, pfzw, &f2_phi, errmsg), errmsg, errmsg);
        if (f1_phi * f2_phi < 0){
          phi_low  = iv_low;
          phi_high = phi;
          found = 1;
          break;
        }
      }
    }
    if (!found){
      class_stop(errmsg, "Couldn't bracket φ in valid intervals [%g,%g] and [%g,%g] params: (%g %g %g %g)", phi_intervals[0][0], phi_intervals[0][1], phi_intervals[1][0], phi_intervals[1][1], xi_smg, v0_smg, m2_smg, lambda_smg);
    }

    /** Find root using Ridders method (Exchange for bisection if you are old-school) */
    class_call(input_fzero_ridder(input_fzerofun_1d,
                                  phi_low,
                                  phi_high,
                                  tol_x_rel*MAX(fabs(phi_low), fabs(phi_high)),
                                  pfzw,
                                  &f1_phi,
                                  &f2_phi,
                                  xzeros,
                                  fevals,
                                  errmsg),
              errmsg,errmsg);
    return _SUCCESS_;
  }

   /** 
   * ==============================
   * ====== Tune shift + Mpl ======
   * ==============================
   */

  else if (unknown_parameters_size == 2){
    /*
    * For each valid phi interval (excluding [phi2_1, phi2_2]):
    * 1) Use winding number to confirm a root exists inside the rectangle
    * 2) Coarse-grain scan to find a good initial guess
    * 3) Newton-Raphson to converge to the root
    */

    double alpha_low  = 1.;
    double alpha_high = 2.;

    // 2 * (winding_lattice_size_alpha + winding_lattice_size_phi) - 4 total points  
    int winding_lattice_size_alpha = 9;
    int winding_lattice_size_phi   = 9;
    // (root_lattice_size_alpha - 1) * (root_lattice_size_phi - 1) 
    int root_lattice_size_alpha    = 7;
    int root_lattice_size_phi      = 7;


    int Niter, MAXIT = 100;
    double xt[2], x1[2], x2[2];
    double F[2], F1[2], F2[2], F_adaptive[2];
    double dphi, dalpha, J11, J12, J21, J22, det;
    double clamp, F_norm, beta;
    double xt_adaptive[2], lastF[2];
    double root_beta_phi, root_beta_alpha;
    double hx, hy;
    int stuck_counter;

    for (int iv = 0; iv < 2; iv++){
      double iv_phi_low  = phi_intervals[iv][0];
      double iv_phi_high = phi_intervals[iv][1];
      if (iv_phi_high <= iv_phi_low) continue;  // interval is empty/invalid

      double root_measure = 100000000.;
      double root_loc[2];
      double winding_number = 0.;

      /*
      * ====================================
      * ===== Calculate winding number =====
      * ====================================
      */
      if (input_verbose >= 3){
        printf("=== Calculating winding number for φ in [%g, %g] ===\n", iv_phi_low, iv_phi_high);
      }

      compute_winding_number_2d(pfzw, iv_phi_low, iv_phi_high, alpha_low, alpha_high,
                                winding_lattice_size_phi, winding_lattice_size_alpha,
                                fevals, root_loc, &root_measure, &winding_number, errmsg);

      if (input_verbose >= 2){
        printf("=== Absolute of winding number: |w| = %g\n\n", winding_number);
      }
      if (winding_number < 0.9){
        if (input_verbose >= 2){
          printf("φ = [%g, %g], α = [%g, %g]: winding number |w| = %g < 0.9, skipping interval.\n", iv_phi_low, iv_phi_high, alpha_low, alpha_high, winding_number);
        }
        continue;
      }

      /*
      * =====================================
      * ===== Coarse grain root finding =====
      * =====================================
      * 
      * As root should exist in this boundary, do some coarse grain calculation to find
      * estimate the location for root.
      */
      find_coarse_root_2d(pfzw, iv_phi_low, iv_phi_high, alpha_low, alpha_high,
                          root_lattice_size_phi, root_lattice_size_alpha,
                          fevals, root_loc, &root_measure, errmsg);

      /*
      * ==========================
      * ===== Newton-Raphson =====
      * ==========================
      */
      xt[0] = root_loc[0];
      xt[1] = root_loc[1];
      beta = 1.;
      stuck_counter = 0;

      for (Niter = 0; Niter < MAXIT; Niter++){
        input_try_unknown_parameters(xt, 2, pfzw, F, errmsg);
        if (Niter > 0){
          if (fabs(lastF[0] - F[0]) < tol_F && fabs(lastF[1] - F[1]) < tol_F){
            stuck_counter++;
            if (stuck_counter == 5){
              class_stop(errmsg, "Could not find root. Algorithm got stuck in φ = [%g, %g], α = [%g, %g]. params (xi, v0, m2, lambda): (%g, %g, %g, %g) w = %g", iv_phi_low, iv_phi_high, alpha_low, alpha_high, xi_smg, v0_smg, m2_smg, lambda_smg, winding_number);
            }
          }
          else {
            stuck_counter = 0;
          }
        }
        memcpy(lastF, F, sizeof(F));
        *fevals += 1;
        if (fabs(F[0]) <= tol_F && fabs(F[1]) <= tol_F){
          xzeros[0] = xt[0];
          xzeros[1] = xt[1];
          if (iv_phi_low <= xzeros[0] && xzeros[0] <= iv_phi_high && alpha_low <= xzeros[1]){
            return _SUCCESS_;
          }
          else{
            class_stop(errmsg, "Solution is not inside the boundary. φ* %g <= %g <= %g, α: %g <= %g <= %g", iv_phi_low, xzeros[0], iv_phi_high, alpha_low, xzeros[1], alpha_high);
          }
        }
        // printf("F = [%g, %g], tol = %g\n", F[0], F[1], tol_F);
        hx = cbrt(DBL_EPSILON) * fmax(fabs(xt[0]), 1.0);
        hy = cbrt(DBL_EPSILON) * fmax(fabs(xt[1]), 1.0);
        x1[0] = xt[0]+hx; x1[1] = xt[1];
        x2[0] = xt[0]-hx; x2[1] = xt[1];
        input_try_unknown_parameters(x1, 2, pfzw, F1, errmsg);
        input_try_unknown_parameters(x2, 2, pfzw, F2, errmsg);
        *fevals += 2;
        J11 = (F1[0] - F2[0]) / (2.*hx);
        J21 = (F1[1] - F2[1]) / (2.*hx);
        x1[0] = xt[0]; x1[1] = xt[1]+hy;
        x2[0] = xt[0]; x2[1] = xt[1]-hy;
        input_try_unknown_parameters(x1, 2, pfzw, F1, errmsg);
        input_try_unknown_parameters(x2, 2, pfzw, F2, errmsg);
        *fevals += 2;
        J12 = (F1[0] - F2[0]) / (2.*hy);
        J22 = (F1[1] - F2[1]) / (2.*hy);
        det = J11*J22 - J12*J21;
        if (fabs(det) < 1e-20){
          class_stop(errmsg, "Determinant is singular. params (xi, v0, m2, lambda): (%g, %g, %g, %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
        }
        dphi   = -(F[0]*J22 - F[1]*J12) / det;
        dalpha = -(F[1]*J11 - F[0]*J21) / det;
        // printf("J = [%g, %g; %g, %g] det = %g\n", J11, J12, J21, J22, det);
        // printf("step: dphi = %g, dalpha = %g\n", dphi, dalpha);
        // printf("xt_low = [%g, %g], xt_high = [%g, %g]\n", iv_phi_low, alpha_low, iv_phi_high, alpha_high);
        // printf("xt = [%g, %g]\n", xt[0], xt[1]);

        F_norm = fabs(F[0]) + fabs(F[1]);
        beta = 1.0;
        for (int k = 0; k < 10; k++){
          xt_adaptive[0] = xt[0] + beta * dphi;
          xt_adaptive[1] = xt[1] + beta * dalpha;

          if (xt_adaptive[0] < iv_phi_low || xt_adaptive[0] > iv_phi_high || xt_adaptive[1] < alpha_low || xt_adaptive[1] > alpha_high){
            root_beta_phi   = (dphi   > 0) ? (iv_phi_high - xt[0]) / (beta * dphi)   : (iv_phi_low  - xt[0]) / (beta * dphi);
            root_beta_alpha = (dalpha > 0) ? (alpha_high  - xt[1]) / (beta * dalpha) : (alpha_low   - xt[1]) / (beta * dalpha);
            clamp = fmin(1., fmin(root_beta_phi, root_beta_alpha));
            xt_adaptive[0] = xt[0] + clamp * beta * dphi;
            xt_adaptive[1] = xt[1] + clamp * beta * dalpha;
          }
          // printf("xt_adaptive = [%g, %g]\n", xt_adaptive[0], xt_adaptive[1]);
          input_try_unknown_parameters(xt_adaptive, 2, pfzw, F_adaptive, errmsg);
          *fevals += 1;
          if (fabs(F_adaptive[0]) + fabs(F_adaptive[1]) < F_norm) break;
          beta *= 0.5;
        }
        xt[0] = xt_adaptive[0];
        xt[1] = xt_adaptive[1];
      }
      class_stop(errmsg, "Could not find solution in φ = [%g, %g]. params (xi, v0, m2, lambda): (%g, %g, %g, %g) w = %g", iv_phi_low, iv_phi_high, xi_smg, v0_smg, m2_smg, lambda_smg, winding_number);
    }
    class_stop(errmsg, "No valid interval had winding number >= 0.9. params: (%g %g %g %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
  }
  else {
    class_stop(errmsg, "Unknown parameter count is bigger than > 2 (%i)\n", unknown_parameters_size);
  }
  class_stop(errmsg, "Could not find solution with high enough accuracy. params (xi, v0, m2, lambda): (%g, %g, %g, %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
  // return _SUCCESS_;
}
