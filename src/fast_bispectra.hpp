#ifndef FAST_BISPECTRA_HEADER_INCLUDED
#define FAST_BISPECTRA_HEADER_INCLUDED

#include "bispectra.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <iostream>

using namespace std;

class fast_bispectra {
private:
  params *para;
  cosmology *cosmo;
  spectra *spec;
  bispectra *bispec;
  int nK, nq, nmu, nphi, nkb;
  int nk_spl, nk1, nk2;
  string kernel_root, fidmodels_dir, fidmodels_config;
  bool verbose, flag_SPT;
  double eta, lambda, c;
  double pi;
  double kmin_fidP0, kmax_fidP0, kmin_spl, kmax_spl, sigmad2_max;
  double Kmin, Kmax, qmin, qmax;
  double k1min, k1max, k2min, k2max;
  double mumin, mumax, phimin, phimax;
  double *L1, *N2, *T3, *U3, *V3;
  double *K, *q, *mu, *phi, *wq, *wmu, *wphi;
  double *Bkfid_1l, *dBk_1l, *Qk;
  gsl_integration_glfixed_table *t_q, *t_mu, *t_phi;
  gsl_spline *spl_fidP0, *spl_sigmad2;
  gsl_interp_accel *acc_fidP0, *acc_sigmad2;
  void find_nearest_fiducial(void);
  void load_K_bin(void);
  void load_linear_power(void);
  void load_kernel(char *base, double *data, int size, int ik);
  void load_kernels_bispec(int ik);
  void set_sigmad2_spline(void);
  double fidP0(double k);
  double sigmad2(double k);
  void compute_fiducial_bispectra(void);
  void compute_delta_bispectra(void);

public:
  fast_bispectra(params &para, cosmology &cosmo, spectra &spec);
  ~fast_bispectra();
  double get_bispectrum(Type a, Type b, Type c, double k1, double k2,
                        double k3);
  vector<double> get_binned_bispectrum(Type a, Type b, Type c,
                                       map<string, vector<double>> kbin);
};

#endif
