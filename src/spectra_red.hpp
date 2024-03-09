#ifndef SPECTRA_RED_HEADER_INCLUDED
#define SPECTRA_RED_HEADER_INCLUDED

#include "cosmology.hpp"
#include "direct_red.hpp"
#include "fast_spectra.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <string>
#include <utility>

using namespace std;

class spectra_red {
private:
  cosmology *cosmo;
  direct_red *dir_red;
  fast_spectra *fast_spec;
  int nmu, ng;
  double pi;
  double eta, f, sigma_v, sigma_vlin, b1, beta, gamma;
  double alpha_perp, alpha_para, rs_drag_ratio;
  double L;
  double b2, bs2, b3nl, N_shot;
  string FoG_type;
  bool flag_AP, flag_fast, flag_SPT, flag_direct_spline, flag_sigma_vlin;
  bool flag_bias, flag_local_Lagrangian_bias;
  gsl_integration_glfixed_table *t_mu;
  double D_FoG(double x);
  double gamma_func(double a, double x);
  double conversion_multipole(int l, int n, double alpha);
  double conversion_wedge(double mu1, double mu2, int n, double alpha);

public:
  spectra_red(params &para, cosmology &cosmo, fast_spectra &fast_spec);
  spectra_red(params &para, cosmology &cosmo, direct_red &dir_red);
  ~spectra_red();
  double get_2D_power(double k, double mu);
  vector<vector<double>> get_2D_power(vector<double> k, vector<double> mu);
  vector<vector<double>> get_multipoles(vector<double> k, vector<int> l);
  vector<vector<double>> get_wedges(vector<double> k, vector<double> wedges);
  pair<vector<double>, vector<vector<double>>>
  get_multipoles_grid(vector<double> kbin, vector<int> l);
  tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
  get_wedges_grid(vector<double> kbin, vector<double> wedges);
};

#endif
