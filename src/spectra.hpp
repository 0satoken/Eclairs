#ifndef SPECTRA_HEADER_INCLUDED
#define SPECTRA_HEADER_INCLUDED

#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <string>

using namespace std;

enum Type_corr2 {
  TREE_TREE = 1,
  TREE_ONELOOP = 2,
  ONELOOP_ONELOOP = 3,
};

class spectra {
private:
  cosmology *cosmo;
  double pi;
  double eta;
  double kmin, kmax, kmin_tab, kmax_tab, kminout, kmaxout;
  int nint, nk_G1_1, nk_G1_2, nk_G2_1, nx, nmu, nkout;
  gsl_integration_glfixed_table *t, *t_G1_1, *t_G1_2, *t_G2_1, *t_x, *t_mu;
  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_monte_vegas_state *s;
  size_t dim, MC_calls;
  double MC_tol;
  double lambda, sigma_d;
  bool flag_free_sigma_d, flag_output, flag_1loop, flag_verbose;
  string model, spacing, output_fname;

public:
  spectra(params &para, cosmology &cosmo);
  ~spectra();
  double get_sigmad2(double k, double lambda);
  double P0(double k);
  double Plin(double k);
  double Pno_wiggle0(double k);
  double Pno_wiggle(double k);
  double Gamma1_1loop(Type a, double k);
  double Gamma1_2loop(Type a, double k);
  double Gamma2_tree(Type a, double k1, double k2, double k3);
  double Gamma2_1loop(Type a, double k1, double k2, double k3);
  double Pcorr2(Type a, Type b, Type_corr2 c, double k);
  double Pcorr2_kernel(Type a, Type b, Type_corr2 c, double k, double q,
                       double mu);
  double Pcorr3(Type a, Type b, double k);
  static double Pcorr3_kernel(double x[], size_t dim, void *p);
  double Preg_1loop(Type a, Type b, double k);
  double Preg_2loop(Type a, Type b, double k);
  double Pspt_1loop(Type a, Type b, double k);
  double Pspt_2loop(Type a, Type b, double k);
  void output_spectra(void);
};

struct Pcorr3_integral_params {
  Type a, b;
  double k, kmin, kmax;
  cosmology *cosmo;
};

#endif
