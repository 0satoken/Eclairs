#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <map>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include "io.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "vector.hpp"
#include "spectra.hpp"


cosmology::cosmology(params &params){
  this->para = &params;
  pi = 4.*atan(1.0);
  e  = exp(1.0);
  T_CMB = 2.726; //[K]
  c_light = 299792.458; //[km/s]

  /* cosmological parameters */
  w_de = -1.0;
  H0 = para->dparams["H0"];
  h = H0/100.0;
  Omega_m = para->dparams["Omega_m"];
  Omega_b = para->dparams["Omega_b"];
  Omega_de = 1.0 - Omega_m;
  ns = para->dparams["ns"];
  As = para->dparams["As"];
  k_pivot = para->dparams["k_pivot"]; //[Mpc^-1]
  fnu = para->dparams["m_nu"]/(93.14*h*h); // neutrino fraction
  z = para->dparams["z"];


  nint = para->iparams["nint"]; // number of integration steps


  /* Flag for RegPT+ */
  if(para->bparams["free_sigma_d"]){
    sigma_d = params.dparams["sigma_d"];
  }

  scale = 1.0/(1.0+z);
  H = H0*sqrt(Omega_m*pow(scale, -3.0)+Omega_de*pow(scale, -3.0*(1.0+w_de)));
  get_eta();
  get_growth_rate();

  /* setting linear power spectrum */
  if(para->bparams["transfer_EH"]){
    set_nw_spectra();
  }
  else if(para->bparams["transfer_from_file"]){
    read_transfer(para->sparams["transfer_file_name"].c_str());
    set_spectra();
  }

  /* smoothed linear power spectrum */
  if(para->bparams["smoothed_linear_power"]){
    set_smoothed_spectra();
  }

}

cosmology::~cosmology(){
  if(flag_transfer){
    delete[] k;
    delete[] Tk;
  }
  if(flag_set_spectra){
    delete[] Plin;
    delete[] P0;
    delete[] Pno_wiggle;
  }
}

/*
 * This function reads transfer function from tabulated file.
 * The table should be CAMB format (7th column corresponds to total components).
 */
void cosmology::read_transfer(const char *transfer_fname){
  ifstream ifs;
  int lines, count;
  double dummy[5];
  string str;

  ifs.open(transfer_fname, ios::in);
  if(ifs.fail()){
    cerr << "[ERROR] transfer file open error:" << transfer_fname << endl;
    exit(1);
  }

  lines = 0;
  while(getline(ifs, str)){
    if(str[0] != '#') lines++;
  }
  nk = lines;

  k = new double[nk];
  Tk = new double[nk];

  ifs.clear();
  ifs.seekg(0, ios_base::beg);

  count = 0;
  while(getline(ifs, str)){
    if(str[0] != '#'){
      sscanf(str.data(), "%lf %lf %lf %lf %lf %lf %lf",
             &k[count], &dummy[0], &dummy[1], &dummy[2], &dummy[3], &dummy[4], &Tk[count]);
      count++;
    }
  }

  for(int i=0;i<nk;i++){
    Tk[i] = sqr(k[i]*h)*Tk[i];
  }

  flag_transfer = true;

  return;
}

/*
 * loading transfer function at z = 0 from given arrays.
 * used from python module
 */
void cosmology::set_transfer(vector<double> k_set, vector<double> Tk_set){
  if(k_set.size() != Tk_set.size()){
      cerr << "[ERROR] The sizes of k and Tk should be the same." << endl;
      exit(1);
  }

  nk = k_set.size();

  k = new double[nk];
  Tk = new double[nk];

  for(int i=0;i<nk;i++){
    /*
    this wavenumber (k) should be in [h/Mpc].
    Note that the wavenumber of transfer function by CAMB
    is in the unit of [1/Mpc] (see CAMB Notes)
    */
    k[i] = k_set[i];
    Tk[i] = Tk_set[i];
  }

  for(int i=0;i<nk;i++){
    Tk[i] = sqr(k[i]*h)*Tk[i];
  }

  flag_transfer = true;

  return;
}

/*
 * This function sets fundamental spectra (linear power spectrum and
 * no-wiggle power spectrum) with the loaded transfer function.
 */
void cosmology::set_spectra(void){
  double Delta_R, Delta_H, T_EH, Dplus, D0;

  if(!flag_transfer){
    cerr << "[ERROR] Transfer function should be loaded first." << endl;
    exit(1);
  }

  Plin = new double[nk];
  P0 = new double[nk];
  Pno_wiggle = new double[nk];

  /* For an explicit formulation for the use of EH transfer function,
   * see Takada et al., PRD, 73, 083520, (2006).
   * We need to care about the transformation of the primordial curvature
   * fluctuation into the matter fluctuation.
   */
  for(int i=0;i<nk;i++){
    Delta_R = As*pow((k[i]*h)/k_pivot, ns-1.0);
    Delta_H = 4.0/25.0*As*pow((k[i]*h)/k_pivot, ns-1.0);
    T_EH = nowiggle_EH_transfer(k[i]);
    Dplus = get_growth_factor(scale);
    D0 = get_growth_factor(1.0);
    Pno_wiggle[i] = (2.0*sqr(pi)/cub(k[i]))*Delta_H*
                    qua((k[i]*h)/(H0/c_light))/sqr(Omega_m)*sqr(T_EH);
    Pno_wiggle[i] *= sqr(D0);
    P0[i] = (2.0*sqr(pi)/cub(k[i]))*Delta_R*sqr(Tk[i]);
    Plin[i] = sqr(Dplus/D0)*P0[i];
  }

  flag_set_spectra = true;

  return;
}

/*
 * This function sets fundamental spectra (linear power spectrum and
 * no-wiggle power spectrum).
 * In this function, no-wiggle spectrum is the input linear power spectrum.
 */
void cosmology::set_nw_spectra(void){
  double Delta_H, T_EH, Dplus, D0;
  double kmin_nw = 1e-5;
  double kmax_nw = 1e3;

  nk = 300;

  k = new double[nk];
  Tk = new double[nk];

  for(int i=0;i<nk;++i){
    k[i] = log(kmin_nw) + (log(kmax_nw)-log(kmin_nw))/((double) nk)*i;
    k[i] = exp(k[i]);
    Tk[i] = 0.0;
  }

  Plin = new double[nk];
  P0 = new double[nk];
  Pno_wiggle = new double[nk];


  for(int i=0;i<nk;i++){
    Delta_H = 4.0/25.0*As*pow((k[i]*h)/k_pivot, ns-1.0);
    T_EH = nowiggle_EH_transfer(k[i]);
    Dplus = get_growth_factor(scale);
    D0 = get_growth_factor(1.0);
    Pno_wiggle[i] = (2.0*sqr(pi)/cub(k[i]))*Delta_H*
                    qua((k[i]*h)/(H0/c_light))/sqr(Omega_m)*sqr(T_EH);
    Pno_wiggle[i] *= sqr(D0);
    P0[i] = Pno_wiggle[i];
    Plin[i] = sqr(Dplus/D0)*Pno_wiggle[i];
  }

  flag_set_spectra = true;

  return;

}

/*
 * Smoothes wiggle feature with Gaussian smoothing in log space.
 * The smoothed spectra replace linear power spectra.
 * For details, see Appedix A of Vlah et al., JCAP, 03(2016)057
 */
void cosmology::set_smoothed_spectra(void){
  int nsm;
  double Dplus, D0, klog, qlogi, qi, wi, lambda, ksmmin, ksmmax, res;
  gsl_interp_accel *acc, *acc_nw;
  gsl_spline *Pspl, *Pspl_nw;
  gsl_integration_glfixed_table *t_sm;

  if(!flag_set_spectra){
    cerr << "[ERROR] Linear power spectrum (with wiggle) should be set first." << endl;
    exit(1);
  }

  /* spline function for linear power spectrum */
  acc = gsl_interp_accel_alloc();
  Pspl = gsl_spline_alloc(gsl_interp_cspline, nk);
  gsl_spline_init(Pspl, k, P0, nk);

  /* spline function for no-wiggle power spectrum */
  acc_nw = gsl_interp_accel_alloc();
  Pspl_nw = gsl_spline_alloc(gsl_interp_cspline, nk);
  gsl_spline_init(Pspl_nw, k, Pno_wiggle, nk);

  Dplus = get_growth_factor(scale);
  D0 = get_growth_factor(1.0);

  nsm = 200;
  ksmmin = 5e-5;
  ksmmax = 50.0;
  lambda = 0.25;

  if(k[0] > ksmmin || k[nk-1] < ksmmax){
    cerr << "[ERROR] the wavenumber range in transfer function";
    cerr << "is too narrow for smoothing" << endl;
    exit(1);
  }

  t_sm = gsl_integration_glfixed_table_alloc(nsm);

  for(int j=0;j<nk;j++){
    klog = log10(k[j]);
    res = 0.0;
    for(int i=0;i<nsm;i++){
      gsl_integration_glfixed_point(log10(ksmmin), log10(ksmmax), i, &qlogi, &wi, t_sm);
      qi = pow(10.0, qlogi);
      res += wi*gsl_spline_eval(Pspl, qi, acc)/gsl_spline_eval(Pspl_nw, qi, acc_nw)
             *exp(-0.5/sqr(lambda)*sqr(klog-qlogi));
    }
    res *= 1.0/sqrt(2.0*pi)/lambda;
    res *= gsl_spline_eval(Pspl_nw, k[j], acc_nw);
    P0[j] = res;
    Plin[j] = sqr(Dplus/D0)*res;
  }


  gsl_spline_free(Pspl);
  gsl_spline_free(Pspl_nw);
  gsl_interp_accel_free(acc);
  gsl_interp_accel_free(acc_nw);
  gsl_integration_glfixed_table_free(t_sm);


  return;
}

/*
 * This function gives growth factor at the given scale factor.
 * The growth factor is not normalized as D(a=1) = 1,
 * but it behaves as D(a) ~ a at high redshift like in EdS Universe.
 */
double cosmology::get_growth_factor(double a){
  double E, D, abserr;
  double p[3] = {Omega_m, Omega_de, w_de};
  gsl_function integrand;
  gsl_integration_workspace *workspace;

  integrand.function = &(cosmology::growth_integrand);
  integrand.params   = p;

  workspace = gsl_integration_workspace_alloc(LIMIT_SIZE);

  gsl_integration_qags(&integrand, 0.0, a, 1e-9, 1e-9, 10000, workspace, &D, &abserr);

  E = sqrt(Omega_m*pow(a, -3.0)+Omega_de*pow(a, -3.0*(1.0+w_de)));
  D *= 5.0/2.0*Omega_m*E;

  gsl_integration_workspace_free(workspace);

  return D;
}

/* The integrand for the growth factor is 1/(a^3 H^3) */
double cosmology::growth_integrand(const double x, void *param){
  double *p;
  double om, ode, w;

  p = (double *) param;
  om  = p[0];
  ode = p[1];
  w   = p[2];
  return pow(om/x+ode*pow(x, -1.0-3.0*w), -3.0/2.0);
}

/*
 * Eisenstein and Hu transfer function fitting formula.
 * k should be in the unit of [h/Mpc].
 * Ref.: Eq [28-31] of Eisenstein and Hu, ApJ, 496, 605, (1998).
 */
double cosmology::nowiggle_EH_transfer(double k){
  double T0, L0, C0, q, theta_CMB, alpha, s, gam_eff;

  s  = 44.5*log(9.83/(Omega_m*h*h))/sqrt(1.0+10.0*pow(Omega_b*h*h, 3.0/4.0));
  s *= h; // Now, s is in the unit of [Mpc/h]
  alpha = 1.0-0.328*log(431.0*Omega_m*h*h)*Omega_b/Omega_m
          +0.38*log(22.3*Omega_m*h*h)*pow(Omega_b/Omega_m, 2.0);
  gam_eff = Omega_m*h*(alpha+(1.0-alpha)/(1.0+pow(0.43*k*s, 4.0)));
  theta_CMB = T_CMB/2.7;

  q  = k*theta_CMB*theta_CMB/gam_eff;
  L0 = log(2.0*e+1.8*q);
  C0 = 14.2+731.0/(1.0+62.5*q);
  T0 = L0/(L0+C0*q*q);

  return T0;
}

double cosmology::nowiggle_EH_power(double k){
    double Delta_H, T_EH, Dplus, Pno_wiggle;

    Delta_H = 4.0/25.0*As*pow((k*h)/k_pivot, ns-1.0);
    T_EH = nowiggle_EH_transfer(k);
    Dplus = get_growth_factor(scale);

    Pno_wiggle = sqr(Dplus)*(2.0*sqr(pi)/cub(k))*Delta_H*
                 qua((k*h)/(H0/c_light))/sqr(Omega_m)*sqr(T_EH);

    return Pno_wiggle;
}

/*
 * Computing sigma8 from the linear power spectrum.
 */
double cosmology::get_sigma8(){
  double R, ki, result;
  double *x, *y;

  if(!flag_set_spectra){
    cerr << "[ERROR] Linear power spectrum should be set first." << endl;
    exit(1);
  }

  R = 8.0; //[Mpc/h]
  x = new double[nk];
  y = new double[nk];

  for(int i=0;i<nk;i++){
    ki   = k[i];
    x[i] = ki;
    y[i] = 1.0/(2.0*sqr(pi))*sqr(ki)*
           sqr(3.0/cub(ki*R)*(sin(ki*R)-ki*R*cos(ki*R)))*P0[i];
  }

  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, nk);

  gsl_spline_init(spline, x, y, nk);
  result = gsl_spline_eval_integ(spline, x[0], x[nk-1], acc);
  result = sqrt(result);

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  delete[] x;
  delete[] y;

  sigma8 = result;

  return result;
}

/*
 * Computing conformal time.
 * eta = log(D), but normalized as eta0 = 0.
 */
double cosmology::get_eta(void){
  eta = log(get_growth_factor(scale)/get_growth_factor(1.0));

  return eta;
}

/*
 * Computing growth rate f = dlnD/dlna
 */
double cosmology::get_growth_rate(void){
  double dDda, E, dE, D;

  D = get_growth_factor(scale);
  E = sqrt(Omega_m/cub(scale)+Omega_de*pow(scale, -3.0*(1.0+w_de)));
  dE = 0.5/E*(-3.0*Omega_m/qua(scale)-
       3.0*(1.0+w_de)*Omega_de*pow(scale, -4.0-3.0*w_de));
  dDda = D*dE/E + 5./2.*Omega_m/sqr(E)/cub(scale);
  f = scale/get_growth_factor(scale)*dDda;

  return f;
}

/*
 * This function computes the comoving distance at the given scale factor.
 * Flat cosmology is assumed. The unit is [Mpc/h].
 */
double cosmology::get_comoving_distance(double a){
  double ai, Ei, wi, res;
  gsl_integration_glfixed_table *t;

  t = gsl_integration_glfixed_table_alloc(nint);
  res = 0.0;

  for(int i=0;i<nint;++i){
    gsl_integration_glfixed_point(a, 1.0, i, &ai, &wi, t);
    Ei = sqrt(Omega_m*pow(ai, -3.0)+Omega_de*pow(ai, -3.0*(1.0+w_de)));
    res += wi/(ai*ai*Ei);
  }

  res *= c_light/100.0;
  gsl_integration_glfixed_table_free(t);

  return res;
}
