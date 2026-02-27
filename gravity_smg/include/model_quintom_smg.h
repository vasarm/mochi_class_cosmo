#include "common.h"
#include "input.h"

int input_find_root_quintom(double *xzeros,
                    int *fevals,
                    int unknown_parameters_size,
                    double tol_x_rel,
                    double tol_F,
                    struct fzerofun_workspace *pfzw,
                    int input_verbose,
                    ErrorMsg errmsg);

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
);

int input_fzero_ridder_quintom(int (*func)(double *x,
                                   int unkown_parameter_size,
                                   void *param,
                                   double *y,
                                   ErrorMsg error_message),
                       int parameter_idx,
                       int unkown_parameter_size,
                       double* x1,
                       double* x2,
                       double xtol,
                       void *param,
                       double *Fx1,
                       double *Fx2,
                       double *xzeros,
                       int *fevals,
                       ErrorMsg error_message);
