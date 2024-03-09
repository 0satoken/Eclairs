#ifndef DIRECT_RED_HEADER_INCLUDED
#define DIRECT_RED_HEADER_INCLUDED

#include "cosmology.hpp"
#include "kernel.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <map>

using namespace std;

struct Aterm_integral_params {
  double k, beta, eta;
  double kmin, kmax, mumin, mumax, phimin, phimax;
  int n;
  bool flag_SPT;
  gsl_interp_accel *acc_sigmad2;
  gsl_spline *spl_sigmad2;
  spectra *spec;
};

class direct_red {
private:
  spectra *spec;
  double pi;
  double f, eta, b1, beta;
  double lambda_p, lambda_b, sigmad2_p_max, sigmad2_b_max;
  double kmin, kmax;
  double mumin, mumax, phimin, phimax;
  double MC_tol;
  int nk_spl, nr, nx, nq, nmu, nphi;
  bool flag_MC, flag_SPT, flag_1loop, flag_spline;
  double *q, *mu, *phi, *wq, *wmu, *wphi;
  gsl_integration_glfixed_table *t_r, *t_x, *t_q, *t_mu, *t_phi;
  gsl_interp_accel *acc_Pk1l_dd, *acc_Pk1l_dt, *acc_Pk1l_tt;
  gsl_interp_accel *acc_sigmad2_p, *acc_sigmad2_b;
  gsl_spline *spl_Pk1l_dd, *spl_Pk1l_dt, *spl_Pk1l_tt;
  gsl_spline *spl_sigmad2_p, *spl_sigmad2_b;
  gsl_spline **spl_Pk, **spl_A, **spl_B, **spl_C;
  gsl_interp_accel **acc_Pk, **acc_A, **acc_B, **acc_C;
  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_monte_vegas_state *s;
  size_t dim, MC_calls;
  void set_all_spline(void);
  void set_Pk1l_spline(void);
  void set_sigmad2_spline(void);
  double Pk1l_dd_spl(double k);
  double Pk1l_dt_spl(double k);
  double Pk1l_tt_spl(double k);
  double sigmad2_p(double k);
  double sigmad2_b(double k);
  map<string, vector<double>> Aterm_Bk211(vector<double> k);
  map<string, vector<double>> Aterm_Bk222_Bk321(vector<double> k);
  static double Aterm_Bk222_Bk321_kernel(double X[], size_t dim, void *param);
  static double Bk222_Bk321_kernel(Type a, Type b, Type c, double k1, double k2,
                                   double k3, Vector qq,
                                   Aterm_integral_params *par);
  double Bk211(Type a, Type b, Type c, double k1, double k2, double k3);
  map<string, vector<double>> Aterm_direct(vector<double> k);
  map<string, vector<double>> Aterm_MC(vector<double> k);
  map<string, vector<double>> Bterm(vector<double> k);
  map<string, vector<double>> Cterm(vector<double> k);

public:
  direct_red(params &para, cosmology &cosmo, spectra &spec);
  ~direct_red();
  double Bispec_1loop(Type a, Type b, Type c, double k1, double k2, double k3);
  double Bispec_tree(Type a, Type b, Type c, double k1, double k2, double k3);
  map<string, vector<double>> get_spectra(vector<double> k);
  map<string, vector<double>> get_Aterm(vector<double> k);
  map<string, vector<double>> get_Bterm(vector<double> k);
  map<string, vector<double>> get_Cterm(vector<double> k);
  map<string, double> get_spectra(double k);
  map<string, double> get_Aterm(double k);
  map<string, double> get_Bterm(double k);
  map<string, double> get_Cterm(double k);
  map<string, vector<double>> get_spl_spectra(vector<double> k);
  map<string, vector<double>> get_spl_Aterm(vector<double> k);
  map<string, vector<double>> get_spl_Bterm(vector<double> k);
  map<string, vector<double>> get_spl_Cterm(vector<double> k);
  map<string, double> get_spl_spectra(double k);
  map<string, double> get_spl_Aterm(double k);
  map<string, double> get_spl_Bterm(double k);
  map<string, double> get_spl_Cterm(double k);
};

double A_func(int a, int b, int c, double r, double x);
double At_func(int a, int b, int c, double r, double x);
double B_func(int a, int b, int c, double r, double x);
double C_func(int a, int b, int c, double r, double x);
double Ct_func(int a, int b, int c, double r, double x);

#endif
