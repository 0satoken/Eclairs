#ifndef FAST_KERNELS_BISPEC_HEADER_INCLUDED
#define FAST_KERNELS_BISPEC_HEADER_INCLUDED

#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <iostream>

using namespace std;

class fast_kernels_bispec {
private:
  params *para;
  spectra *spec;
  int myrank, numprocs;
  int nK, nq, nmu, nphi;
  string kernel_root;
  double pi;
  double Kmin, Kmax, qmin, qmax, mumin, mumax, phimin, phimax;
  double *K, *q, *mu, *phi, *wq, *wmu, *wphi;
  double *L1, *M1, *N2, *T3, *U3, *V3;
  gsl_integration_glfixed_table *t_q, *t_mu, *t_phi;
  double fidP0(double k);

public:
  fast_kernels_bispec(params &params, spectra &spec, int myrank, int numprocs);
  ~fast_kernels_bispec();
  void set_K_bin(void);
  void compute_diagram_bispectrum(void);
  void compute_kernels(void);
  void save_linear_power(void);
  void save_kernel_data(char *base, double *data, int size, int ik);
  void L1_kernel(void);
  void N2_kernel(void);
  void T3U3V3_kernel(void);
};

#endif
