#include "model_quintom_smg.h"


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
  double phi_low, phi_high, f1_phi, f2_phi;
  double alpha1, alpha2, f1_alpha, f2_alpha;
  int iter, iter2;
  int return_function;

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
  phi_low = 0.;

  class_test(m2_smg*m2_smg - 4.*lambda_smg*v0_smg < 0, errmsg, "(m^2)^2 - 4 lambda v0 < 0 -> Solution does not exist with these parameters. params: (%g %g %g %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
  
  phi_high = -1./pow(fabs(xi_smg), 0.5) + pow((-m2_smg - pow(m2_smg*m2_smg - 4.*lambda_smg*v0_smg, 0.5))/lambda_smg, 0.5);
  if (input_verbose >= 2){
    printf("phi_min: %g, phi_max: %g\n", phi_low, phi_high);
  }
  
  /** 
   * =============================
   * ====== Tune only shift ======
   * =============================
   */
  if (unknown_parameters_size == 1){
    // NB! Assume this is scalar field then!
    class_call(input_fzerofun_1d(phi_low, pfzw, &f1_phi, errmsg),
              errmsg,
              errmsg);
    int N_tries = 10;
    double phi;
    double dphi = (phi_high-phi_low)/N_tries;
    for (int i=1; i<=N_tries; i++){
      phi = phi_low+i*dphi;
      class_call(input_fzerofun_1d(phi, pfzw, &f2_phi, errmsg),
              errmsg,
              errmsg);
      
      if (f1_phi*f2_phi < 0){
        break;
      }
    }
    phi_high=phi;
    if (f1_phi*f2_phi>0){
      class_stop(errmsg, "Couldn't bracket between φ: [0, %g] -> [%g, %g] params: (%g %g %g %g)", phi_high, f1_phi, f2_phi, xi_smg, v0_smg, m2_smg, lambda_smg);
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
    * 1) Make sure that root exists
    * 2) Find good estimation for the root
    * 3) Find the root
    *
    * 1) Use winding number to determine if root can exist inside the boundaries
    *   1.1) If winding number |w| > 0.9 -> root exists  
    * 2) Use coarse-grain to find good estimation for the root
    * 3) Use some root method to converge to correct root
    */

    int Niter,MAXIT=100;
    
    double alpha_low = 1.;
    double alpha_high = 2.;
    double dx_alpha = 0.2;

    double x1[2], x2[2], xt[2];
    double F1[2], F2[2], F[2], Fl[2], Fh[2];

    int winding_lattice_size_alpha = 9;
    int winding_lattice_size_phi = 9;
    int winding_N = 2 * (winding_lattice_size_alpha + winding_lattice_size_phi) - 4;
    int winding_idx = 0;
    double winding_boundary_F[winding_N][2];
    double winding_boundary_F_cross;
    double winding_boundary_F_dot;
    double winding_number = 0.;
    double winding_step_x, winding_step_y;
    winding_step_x = (phi_high-phi_low)/(winding_lattice_size_phi-1);
    winding_step_y = (alpha_high-alpha_low)/(winding_lattice_size_alpha-1);


    // NB! Make sure that (winding_lattice_size_alpha-1)/(root_lattice_size_alpha-1) dividing exactly
    // 2 Boundary points and 5 inner points
    int root_lattice_size_alpha = 7;
    // NB! Make sure that (winding_lattice_size_phi-1)/(root_lattice_size_phi-1) dividing exactly
    // 2 Boundary points and 5 inner points
    int root_lattice_size_phi = 7;

    double root_step_phi = (phi_high-phi_low)/(root_lattice_size_phi-1);
    double root_step_alpha = (alpha_high-alpha_low)/(root_lattice_size_alpha-1);
    double root_measure2 = 1000000.;
    double measure2;
    double root_loc[2];
    double root_beta;
    double root_beta_phi;
    double root_beta_alpha;

    

    /*
    * ====================================
    * ===== Calculate winding number =====
    * ====================================
    */
   
    if (input_verbose >= 3){
      printf("=== Calculating winding number ===\n");
    }

    // Bottom edge
    xt[1] = alpha_low;
    for (int i=0; i<winding_lattice_size_phi; i++){
      xt[0] = phi_low+i*winding_step_x;
      input_try_unknown_parameters(xt, 2, pfzw, winding_boundary_F[winding_idx], errmsg);
      measure2 = pow(winding_boundary_F[winding_idx][0], 2.) + pow(winding_boundary_F[winding_idx][1], 2.);
      if (measure2 < root_measure2){
        root_measure2 = measure2;
        root_loc[0] = xt[0];
        root_loc[1] = xt[1];
      }

      winding_idx++;
      (*fevals)++;
    }

    // Right edge
    xt[0] = phi_high;
    for (int i=1; i<winding_lattice_size_alpha; i++){
      xt[1] = alpha_low+i*winding_step_y;
      input_try_unknown_parameters(xt, 2, pfzw, winding_boundary_F[winding_idx], errmsg);
      measure2 = pow(winding_boundary_F[winding_idx][0], 2.) + pow(winding_boundary_F[winding_idx][1], 2.);
      if (measure2 < root_measure2){
        root_measure2 = measure2;
        root_loc[0] = xt[0];
        root_loc[1] = xt[1];
      }


      winding_idx++;
      (*fevals)++;
    }

    xt[1] = alpha_high;
    for (int i=1; i<winding_lattice_size_phi; i++){
      xt[0] = phi_high - i*winding_step_x;
      input_try_unknown_parameters(xt, 2, pfzw, winding_boundary_F[winding_idx], errmsg);
      measure2 = pow(winding_boundary_F[winding_idx][0], 2.) + pow(winding_boundary_F[winding_idx][1], 2.);
      if (measure2 < root_measure2){
        root_measure2 = measure2;
        root_loc[0] = xt[0];
        root_loc[1] = xt[1];
      }

      
      winding_idx++;
      (*fevals)++;
    }

    xt[0] = phi_low;
    for (int i=1; i<winding_lattice_size_alpha-1; i++){
      xt[1] = alpha_high - i*winding_step_y;
      input_try_unknown_parameters(xt, 2, pfzw, winding_boundary_F[winding_idx], errmsg);
      measure2 = pow(winding_boundary_F[winding_idx][0], 2.) + pow(winding_boundary_F[winding_idx][1], 2.);
      if (measure2 < root_measure2){
        root_measure2 = measure2;
        root_loc[0] = xt[0];
        root_loc[1] = xt[1];
      }

      winding_idx++;
      (*fevals)++;
    }

    int ip;

    for (int i = 0; i < winding_N; i++) {
      ip = (i+1) % winding_N;
      winding_boundary_F_cross = winding_boundary_F[i][0] * winding_boundary_F[ip][1] - winding_boundary_F[i][1]*winding_boundary_F[ip][0];
      winding_boundary_F_dot = winding_boundary_F[i][0] * winding_boundary_F[ip][0] + winding_boundary_F[i][1]*winding_boundary_F[ip][1];
      winding_number += atan2(winding_boundary_F_cross, winding_boundary_F_dot);
    }

    winding_number = fabs(winding_number)/(2.*_PI_);
    
    if (input_verbose >= 3){
      printf("=== Absolute of winding number: |w| = %g\n\n", winding_number);
    }
    if (winding_number < 0.9){
      class_stop(errmsg, "φ = [%g %g] and α = [%g %g]. Winding number |w| = %g < 0.9. params: (%g %g %g %g)", phi_low, phi_high, alpha_low, alpha_high, winding_number, xi_smg, v0_smg, m2_smg, lambda_smg);
    }
    /*
    * =====================================
    * ===== Coarse grain root finding =====
    * =====================================
    */

    // Don't have to check boundary, as this is already done when calculating boundaries
    for (int i=1; i<root_lattice_size_phi-1; i++){
      for (int j=1; j<root_lattice_size_alpha-1; j++){
        xt[0] = phi_low + root_step_phi*i;
        xt[1] = alpha_low + root_step_alpha*j;
        input_try_unknown_parameters(xt, 2, pfzw, F, errmsg);
        (*fevals)++;

        measure2 = pow(F[0], 2.) + pow(F[1], 2.);
        if (measure2 < root_measure2){
          root_measure2 = measure2;
          root_loc[0] = xt[0];
          root_loc[1] = xt[1];
        }
      }
    }
   
     /*
    * ==========================
    * ===== Newton-Raphson =====
    * ==========================
    */

    double hx, hy;
    xt[0] = root_loc[0];
    xt[1] = root_loc[1];

    double dphi, dalpha;
    double J11, J12, J21, J22, det;
    for (Niter=0; Niter<MAXIT; Niter++){
      input_try_unknown_parameters(xt, 2, pfzw, F, errmsg);
      *fevals += 1;
      if (fabs(F[0]) <= tol_F & fabs(F[1]) <= tol_F){
        xzeros[0] = xt[0];
        xzeros[1] = xt[1];
        if (phi_low <= xzeros[0] && xzeros[0] <= phi_high && alpha_low <= xzeros[1]){
          return _SUCCESS_;
        }
        else{
          class_stop(errmsg, "Solution is not inside the boundary. φ* %g <= %g <= %g, α: %g <= %g <= %g", phi_low, xzeros[0], phi_high, alpha_low, xzeros[1], alpha_high);
        }
      }
      hx = cbrt(DBL_EPSILON) * fmax(fabs(xt[0]), 1.0);
      hy = cbrt(DBL_EPSILON) * fmax(fabs(xt[1]), 1.0);
      x1[0] = xt[0]+hx;
      x1[1] = xt[1];
      x2[0] = xt[0]-hx;
      x2[1] = xt[1];
      input_try_unknown_parameters(x1, 2, pfzw, F1, errmsg);
      input_try_unknown_parameters(x2, 2, pfzw, F2, errmsg);
      *fevals += 2;
      J11 = (F1[0] - F2[0])/(2.*hx);
      J21 = (F1[1] - F2[1])/(2.*hx);
      x1[0] = xt[0];
      x1[1] = xt[1]+hy;
      x2[0] = xt[0];
      x2[1] = xt[1]-hy;
      input_try_unknown_parameters(x1, 2, pfzw, F1, errmsg);
      input_try_unknown_parameters(x2, 2, pfzw, F2, errmsg);
      *fevals += 2;
      J12 = (F1[0] - F2[0])/(2.*hy);
      J22 = (F1[1] - F2[1])/(2.*hy);
      det = J11*J22 - J12*J21;
      if (fabs(det) < 1e-20){
        class_stop(errmsg, "Determinant is singular. params: (%g, %g, %g, %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
      }
      dphi = -(F[0]*J22 - F[1]*J12)/det;
      dalpha = -(F[1]*J11 - F[0]*J21)/det;

      xt[0] = xt[0] + dphi;
      xt[1] = xt[1] + dalpha;
      // NB: Make sure the variables stay in the boundaries!
      // Find highest constant such that step stays inside the boundaries
      if (xt[0] < phi_low || xt[0] > phi_high || xt[1] < alpha_low || xt[1] > alpha_high){
        // Make step back
        xt[0] = xt[0] - dphi;
        xt[1] = xt[1] - dalpha;

        root_beta_phi = (dphi>0) ? (phi_high - xt[0])/dphi : (phi_low - xt[0])/dphi;
        root_beta_alpha = (dalpha>0) ? (alpha_high - xt[1])/dalpha : (alpha_low - xt[1])/dalpha;

        root_beta = fmin(1., fmin(root_beta_phi, root_beta_alpha));
        xt[0] = xt[0] + root_beta*dphi;
        xt[1] = xt[1] + root_beta*dalpha;
      }
    }    
  }
  else {
    class_stop(errmsg, "Unknown parameter count is bigger than > 2 (%i)\n", unknown_parameters_size);
  }
  class_stop(errmsg, "Could not find solution with high enough accuracy. params: (%g, %g, %g, %g)", xi_smg, v0_smg, m2_smg, lambda_smg);
  // return _SUCCESS_;
}

int solve_for_alpha_quintom(
  int unknown_parameters_size,
  double *x1,
  double *x2,
  double xtol,
  void *param,
  double *Fx1,
  double *Fx2,
  double *xzeros,
  int *fevals,
  ErrorMsg error_message
){
  int j,MAXIT=1000;
  double alpha = 1.137970;
  double *fh,*fl,*xh,*xl;
  double Fzero[2] = {0., 0.};

  class_alloc(fh, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(fl, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(xh, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(xl, unknown_parameters_size*sizeof(double), error_message);

  
  if ((Fx1!=NULL)&&(Fx2!=NULL)){
    for (int i=0; i<unknown_parameters_size; i++){
      fl[i] = Fx1[i];
      fh[i] = Fx2[i];
      xl[i] = x1[i];
      xh[i] = x2[i];
    }
  }
  xl[1] = alpha;
  xh[1] = alpha;
  xzeros[1] = alpha;

  input_try_unknown_parameters(xl, unknown_parameters_size, param, fl, error_message);
  // printf("Value 1: %g, Value 2: %g\n", xl[0], xl[1]);
  input_try_unknown_parameters(xh, unknown_parameters_size, param, fh, error_message);
  // printf("Value 1: %g, Value 2: %g\n", xh[0], xh[1]);
  input_fzero_ridder_quintom(
      input_try_unknown_parameters,
      unknown_parameters_size,
      0,
      xl,
      xh,
      xtol,
      param,
      fl,
      fh,
      xzeros,
      fevals,
      error_message
    );
  free(fh); free(fl); free(xh); free(xl);
  // printf("x[0]: %g f[0]: %g, x[1]: %g f[1]: %g.\n", xzeros[0], Fzero[0], xzeros[1], Fzero[1]);
  input_try_unknown_parameters(xzeros, unknown_parameters_size, param, Fzero, error_message);
  // printf("x[0]: %g f[0]: %g, x[1]: %g f[1]: %g.\n", xzeros[0], Fzero[0], xzeros[1], Fzero[1]);
}


int input_fzero_ridder_quintom(int (*func)(double* x,
                                   int unknown_parameters_size,
                                   void *param,
                                   double *y,
                                   ErrorMsg error_message),
                       int unknown_parameters_size,
                       int parameter_idx,
                       double *x1,
                       double *x2,
                       double xtol,
                       void *param,
                       double *Fx1,
                       double *Fx2,
                       double *xzero,
                       int *fevals,
                       ErrorMsg error_message){

  /** Summary: */

  /** Define local variables */
  int j,MAXIT=1000;
  double *ans,*fh,*fl,*fm,*fnew,*xh,*xl,*xm,*xnew;
  double s;
  
  class_alloc(ans, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(fh, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(fl, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(fm, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(fnew, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(xh, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(xl, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(xm, unknown_parameters_size*sizeof(double), error_message);
  class_alloc(xnew, unknown_parameters_size*sizeof(double), error_message);
  
  // Initialization
  for (int i=0; i<unknown_parameters_size; i++){
    xl[i] = x1[i];
    xh[i] = x2[i];
    xm[i] = x2[i];
    xnew[i] = x2[i];
    ans[i] = x2[i];
  }

  if ((Fx1!=NULL)&&(Fx2!=NULL)){
    for (int i=0; i<unknown_parameters_size; i++){
      fl[i] = Fx1[i];
      fh[i] = Fx2[i];
    }
  }
  else{
    class_call((*func)(x1, unknown_parameters_size, param, fl, error_message),
               error_message, error_message);
    class_call((*func)(x2, unknown_parameters_size, param, fh, error_message),
               error_message, error_message);

    *fevals = (*fevals)+2;
  }


  if ((fl[parameter_idx] > 0.0 && fh[parameter_idx] < 0.0) || (fl[parameter_idx] < 0.0 && fh[parameter_idx] > 0.0)) {
    xl[parameter_idx]=x1[parameter_idx];
    xh[parameter_idx]=x2[parameter_idx];
    ans[parameter_idx]=-1.11e11;
    for (j=1;j<=MAXIT;j++) {
      xm[parameter_idx]=0.5*(xl[parameter_idx]+xh[parameter_idx]);
      class_call((*func)(xm, unknown_parameters_size, param, fm, error_message),
                 error_message, error_message);
      *fevals = (*fevals)+1;
      s=sqrt(fm[parameter_idx]*fm[parameter_idx]-fl[parameter_idx]*fh[parameter_idx]);
      if (s == 0.0){
        xzero[parameter_idx] = ans[parameter_idx];
        return _SUCCESS_;
      }
      xnew[parameter_idx]=xm[parameter_idx]+(xm[parameter_idx]-xl[parameter_idx])*((fl[parameter_idx] >= fh[parameter_idx] ? 1.0 : -1.0)*fm[parameter_idx]/s);
      if (fabs(xnew[parameter_idx]-ans[parameter_idx]) <= xtol) {
        xzero[parameter_idx] = ans[parameter_idx];
        return _SUCCESS_;
      }
      
      ans[parameter_idx]=xnew[parameter_idx];
      class_call((*func)(ans, unknown_parameters_size, param, fnew, error_message),
                 error_message, error_message);
                 *fevals = (*fevals)+1;
      if (fnew[parameter_idx] == 0.0){
        xzero[parameter_idx] = ans[parameter_idx];
        return _SUCCESS_;
      }

      if (NRSIGN(fm[parameter_idx],fnew[parameter_idx]) != fm[parameter_idx]) {
        xl[parameter_idx]=xm[parameter_idx];
        fl[parameter_idx]=fm[parameter_idx];
        xh[parameter_idx]=ans[parameter_idx];
        fh[parameter_idx]=fnew[parameter_idx];
      }
      else if (NRSIGN(fl[parameter_idx],fnew[parameter_idx]) != fl[parameter_idx]) {
        xh[parameter_idx]=ans[parameter_idx];
        fh[parameter_idx]=fnew[parameter_idx];
      }
      else if (NRSIGN(fh[parameter_idx],fnew[parameter_idx]) != fh[parameter_idx]) {
        xl[parameter_idx]=ans[parameter_idx];
        fl[parameter_idx]=fnew[parameter_idx];
      }
      else{
        free(ans);free(fh);free(fl);free(fm);free(fnew);free(xh);free(xl);free(xm);free(xnew);
        return _FAILURE_;
      }
      if (fabs(xh[parameter_idx]-xl[parameter_idx]) <= xtol) {
        xzero[parameter_idx] = ans[parameter_idx];
        free(ans);free(fh);free(fl);free(fm);free(fnew);free(xh);free(xl);free(xm);free(xnew);
        return _SUCCESS_;
      }
    }
    class_stop(error_message,"zriddr exceed maximum iterations");
  }

  else {
    if (fl[parameter_idx] == 0.0){
      for (int i = 0; i<unknown_parameters_size; i++){
        xzero[i] = x1[i];
      }
      free(ans);free(fh);free(fl);free(fm);free(fnew);free(xh);free(xl);free(xm);free(xnew);

    } return _SUCCESS_;
    if (fh[parameter_idx] == 0.0){
      for (int i = 0; i<unknown_parameters_size; i++){
        xzero[i] = x2[i];
      }
      free(ans);free(fh);free(fl);free(fm);free(fnew);free(xh);free(xl);free(xm);free(xnew);

      return _SUCCESS_;
    };
    class_stop(error_message,"root must be bracketed in zriddr.");
  }
  class_stop(error_message,"Failure in int.");
}
