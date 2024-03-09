#ifndef FAST_SPECTRA_HEADER_INCLUDED
#define FAST_SPECTRA_HEADER_INCLUDED

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

class fast_spectra {
private:
  params *para;
  cosmology *cosmo;
  spectra *spec;
  int nk, nq, np, nmu, nphi, nr, nx;
  int nk_spl, nk1, nk2;
  string kernel_root, fidmodels_dir, fidmodels_config;
  bool verbose, direct_Bterm, flag_SPT;
  double eta, f, c, beta, b1;
  double pi;
  double lambda_p, lambda_b, sigmad2_p_max, sigmad2_b_max;
  double kmin_fidP0, kmax_fidP0, kmin_spl, kmax_spl;
  double kmin, kmax, qmin, qmax, pmin, pmax;
  double k1min, k1max, k2min, k2max;
  double mumin, mumax, phimin, phimax;
  double *k, *q, *p, *mu, *phi, *wq, *wp, *wmu, *wphi;
  double *L1, *M1, *X2, *Y2, *Z2, *Q2, *R2, *S3;
  double *N2, *T3, *U3, *V3;
  double *Pkfid_1l, *Pkfid_2l, *Bkfid_1l;
  double *dPk_1l, *dPk_2l, *dBk_1l;
  double *A, *B;
  gsl_integration_glfixed_table *t_q, *t_p, *t_mu, *t_phi, *t_r, *t_x;
  gsl_spline *spl_fidP0, *spl_sigmad2_p, *spl_sigmad2_b;
  gsl_spline **spl_Pk_1l, **spl_Pk_2l, **spl_A, **spl_B;
  gsl_interp_accel *acc_fidP0, *acc_sigmad2_p, *acc_sigmad2_b;
  gsl_interp_accel **acc_Pk_1l, **acc_Pk_2l, **acc_A, **acc_B;
  void find_nearest_fiducial(void);
  void load_k_bin(void);
  void load_linear_power(void);
  void load_kernel(char *base, double *data, int size, int ik);
  void load_kernels_spec(int ik);
  void load_kernels_bispec(int ik);
  void set_sigmad2_spline(void);
  double fidP0(double k);
  double sigmad2_p(double k);
  double sigmad2_b(double k);
  void compute_fiducial_spectra(void);
  void compute_fiducial_bispectra(void);
  void compute_delta_spectra(void);
  void compute_delta_bispectra(void);
  void Aterm_recon(void);
  void Bterm_recon(void);
  void construct_spline_spectra(void);
  void construct_spline_Aterm(void);
  void construct_spline_Bterm(void);
  double A_func(int a, int b, int c, double r, double x);
  double At_func(int a, int b, int c, double r, double x);
  double B_func(int a, int b, int c, double r, double x);

public:
  fast_spectra(params &para, cosmology &cosmo, spectra &spec);
  ~fast_spectra();
  map<string, vector<double>> get_spectra_1l(vector<double> k0);
  map<string, vector<double>> get_spectra_2l(vector<double> k0);
  map<string, vector<double>> get_Aterm(vector<double> k0);
  map<string, vector<double>> get_Bterm(vector<double> k0);
  map<string, double> get_spectra_1l(double k0);
  map<string, double> get_spectra_2l(double k0);
  map<string, double> get_Aterm(double k0);
  map<string, double> get_Bterm(double k0);
};

#endif
