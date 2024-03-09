#ifndef IR_EFT_HEADER_INCLUDED
#define IR_EFT_HEADER_INCLUDED

#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

class IR_EFT {
private:
  params *para;
  cosmology *cosmo;
  bool flag_AP;
  int ng, nk_spl, nsm, nq, nr, nx, nmuint;
  double pi;
  double eta, f, b1, beta, L;
  double alpha_perp, alpha_para, rs_drag_ratio;
  double kS, rs;
  double c0, c2, c4, cd4, Pshot;
  double kmin, kmax, ksmmin, ksmmax, lambda;
  double Sigma2, deltaSigma2;
  gsl_integration_glfixed_table *t_r, *t_x, *t_mu;
  gsl_interp_accel *acc_w, *acc_nw;
  gsl_spline *Pspl_w, *Pspl_nw;
  gsl_interp_accel **acc_Pk_1l_nw, **acc_Pk_1l_w;
  gsl_spline **spl_Pk_1l_nw, **spl_Pk_1l_w;
  void set_smoothed_spectra(void);
  void calc_Sigma(void);
  void construct_spline_Pk(void);
  double Z1(Vector k1);
  double Z2(Vector k1, Vector k2);
  double Z3(Vector k1, Vector k2, Vector k3);
  double Z3_unsym(Vector k1, Vector k2, Vector k3);
  double K2(double k, double q, double x);
  double K3(double k, double q);
  double A_func(int n, int m, double r, double x);
  double B_func(int n, int m, double r);

public:
  IR_EFT(params &para, cosmology &cosmo, spectra &spec);
  ~IR_EFT();
  double P_w(double k);
  double P_nw(double k);
  vector<vector<double>> calc_Pkmu_1l(vector<double> k, vector<double> mu);
  double calc_Pkmu_1l(double k, double mu);
  vector<vector<double>> get_multipoles(vector<double> k, vector<int> l);
  vector<vector<double>> get_wedges(vector<double> k, vector<double> wedges);
  pair<vector<double>, vector<vector<double>>>
  get_multipoles_grid(vector<double> kbin, vector<int> l);
  tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
  get_wedges_grid(vector<double> kbin, vector<double> wedges);
};

#endif
