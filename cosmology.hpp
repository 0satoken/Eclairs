#ifndef COSMOLOGY_HEADER_INCLUDED
#define COSMOLOGY_HEADER_INCLUDED

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include "io.hpp"

#define LIMIT_SIZE 10000

using namespace std;

class cosmology{
private:
  int nint;
  double pi, e, T_CMB, c_light;
  static double growth_integrand(const double x, void *param);
public:
  cosmology(params &params);
  ~cosmology();
  params *para;
  bool flag_transfer, flag_set_spectra;
  double Omega_m, Omega_b, Omega_de, w_de, H, H0, h, fnu;
  double As, k_pivot, ns, sigma8;
  double z, scale, eta, f;
  double sigma_d;
  double *k, *Tk, *Plin, *P0, *Pno_wiggle;
  int nk;
  void read_transfer(const char *transfer_fname);
  void set_transfer(vector<double> k_set, vector<double> Tk_set);
  void set_spectra(void);
  void set_nw_spectra(void);
  void set_smoothed_spectra(void);
  double nowiggle_EH_transfer(double k);
  double nowiggle_EH_power(double k);
  double get_growth_factor(double a);
  double get_sigma8(void);
  double get_eta(void);
  double get_growth_rate(void);
  double get_comoving_distance(double a);
};

#endif
