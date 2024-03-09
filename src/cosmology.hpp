#ifndef COSMOLOGY_HEADER_INCLUDED
#define COSMOLOGY_HEADER_INCLUDED

#include "params.hpp"
#include <cmath>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

class cosmology {
private:
  int nint;
  double pi, e, T_CMB, c_light;
  gsl_integration_glfixed_table *t;
  bool flag_transfer, flag_set_spectra;
  double Omega_m, Omega_b, Omega_k, Omega_de, w_de, H, H0, h, fnu;
  double As, k_pivot, ns, sigma8;
  double z, scale, D0, eta, f, chi;
  double kmin_Tk, kmax_Tk, P0min_Tk, Pnw0min_Tk;
  vector<double> k, Tk;
  int nk;
  gsl_interp_accel *acc, *acc_nw;
  gsl_spline *Pspl, *Pspl_nw;
  void read_transfer(const char *transfer_fname);
  void set_nw_spectra(void);
  double nowiggle_EH_transfer(double k);
  double nowiggle_EH_power(double k);

public:
  cosmology(params &params);
  ~cosmology();
  void set_transfer(vector<double> k_set, vector<double> Tk_set);
  void set_spectra(void);
  double get_growth_factor(double a);
  double get_eta(double a);
  double get_growth_rate(double a);
  double get_comoving_distance(double a);
  double get_sigmaR(double R, double a);
  double get_growth_factor(void);
  double get_eta(void);
  double get_growth_rate(void);
  double get_comoving_distance(void);
  double get_linear_displacement_dispersion(void);
  double get_sigma8(void);
  double get_Hubble(void);
  double P0(double k);
  double Plin(double k);
  double Pno_wiggle0(double k);
  double Pno_wiggle(double k);
};

#endif
