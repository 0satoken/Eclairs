#ifndef FAST_KERNELS_HEADER_INCLUDED
#define FAST_KERNELS_HEADER_INCLUDED

#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <iostream>

using namespace std;

class fast_kernels {
private:
  params *para;
  spectra *spec;
  int myrank, numprocs;
  int nk, nq, np, nmu, nphi, nr, nx;
  string kernel_root;
  double pi;
  double qmin, qmax, pmin, pmax;
  double mumin, mumax, phimin, phimax;
  double *k, *q, *p, *mu, *phi, *wq, *wp, *wmu, *wphi;
  double *L1, *M1, *X2, *Y2, *Z2, *Q2, *R2, *S3;
  double *N2, *T3, *U3, *V3;
  gsl_integration_glfixed_table *t_q, *t_p, *t_mu, *t_phi, *t_r, *t_x;
  double fidP0(double k);

public:
  fast_kernels(params &params, spectra &spec, int myrank, int numprocs);
  ~fast_kernels();
  void set_k_bin(void);
  void compute_diagram_spectrum(void);
  void compute_diagram_bispectrum(void);
  void compute_kernels(void);
  void save_linear_power(void);
  void save_kernel_data(char *base, double *data, int size, int ik);
  void L1_kernel(void);
  void M1_kernel(void);
  void X2Y2Z2_kernel(void);
  void Q2R2_kernel(void);
  void S3_kernel(void);
  void N2_kernel(void);
  void T3U3V3_kernel(void);
};

#endif
