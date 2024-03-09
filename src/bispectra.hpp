#ifndef BISPECTRA_HEADER_INCLUDED
#define BISPECTRA_HEADER_INCLUDED

#include "cosmology.hpp"
#include "kernel.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <iostream>

using namespace std;

class bispectra {
private:
  spectra *spec;
  double pi;
  double eta, lambda;
  double kmin, kmax, sigmad2_max;
  double mumin, mumax, phimin, phimax;
  int nk_spl, nq, nmu, nphi;
  bool flag_SPT;
  double *q, *mu, *phi, *wq, *wmu, *wphi;
  gsl_integration_glfixed_table *t_q, *t_mu, *t_phi;
  gsl_interp_accel *acc_sigmad2;
  gsl_spline *spl_sigmad2;

public:
  bispectra(params &para, cosmology &cosmo, spectra &spec);
  ~bispectra();
  void set_sigmad2_spline(void);
  double sigmad2_spl(double k);
  double Bispec_1loop(Type a, Type b, Type c, double k1, double k2, double k3);
  double Bispec_tree(Type a, Type b, Type c, double k1, double k2, double k3);
};

#endif
