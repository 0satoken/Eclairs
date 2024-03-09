#ifndef NONLINEAR_HEADER_INCLUDED
#define NONLINEAR_HEADER_INCLUDED

#include "cosmology.hpp"
#include "vector.hpp"
#include <cmath>
#include <iostream>

using namespace std;

class nonlinear {
private:
  cosmology *cosmo;
  double pi;
  double z, eta, sigma8;
  double h, w_de, Omega_m, Omega_de, fnu, om_m, om_v;
  double kmin, kmax;
  double rn, rncur, ksigma;
  size_t nint, iter_max;

public:
  nonlinear(params &params, cosmology &cosmo);
  ~nonlinear();
  double Plinz(double k);
  void find_ksigma(void);
  void set_omega(void);
  double Pk_halofit(double k);
  double Bk_halofit(double k1, double k2, double k3);
  double F2(double k1, double k2, double k3);
  void change_redshift(double z);
};

#endif
