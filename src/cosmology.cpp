#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

cosmology::cosmology(params &params) {
  pi = 4.0 * atan(1.0);
  e = exp(1.0);
  T_CMB = 2.726;        //[K]
  c_light = 299792.458; //[km/s]

  /* cosmological parameters */
  w_de = params.dparams["w_de"];
  H0 = params.dparams["H0"];
  h = H0 / 100.0;
  Omega_m = params.dparams["Omega_m"];
  Omega_b = params.dparams["Omega_b"];
  Omega_k = params.dparams["Omega_k"];
  Omega_de = 1.0 - Omega_m - Omega_k;
  ns = params.dparams["ns"];
  As = params.dparams["As"];
  k_pivot = params.dparams["k_pivot"];            // [Mpc^-1]
  fnu = params.dparams["m_nu"] / (93.14 * h * h); // neutrino fraction
  z = params.dparams["z"];

  nint = params.iparams["nint"]; // number of steps for generic integration
  t = gsl_integration_glfixed_table_alloc(nint);

  scale = 1.0 / (1.0 + z);
  H = H0 * sqrt(Omega_m / cub(scale) + Omega_k / sqr(scale) +
                Omega_de * pow(scale, -3.0 * (1.0 + w_de)));

  /* setting linear power spectrum */
  if (params.bparams["transfer_EH"]) {
    set_nw_spectra();
  } else if (params.bparams["transfer_from_file"]) {
    read_transfer(params.sparams["transfer_file_name"].c_str());
    set_spectra();
  }

  D0 = get_growth_factor(scale);
  eta = get_eta(scale);
  f = get_growth_rate(scale);
  chi = get_comoving_distance(scale);
  // sigma8 = get_sigmaR(8.0, 1.0); // usual sigma8 (z=0)
}

cosmology::~cosmology() {
  gsl_integration_glfixed_table_free(t);
  if (flag_set_spectra) {
    gsl_spline_free(Pspl);
    gsl_spline_free(Pspl_nw);
    gsl_interp_accel_free(acc);
    gsl_interp_accel_free(acc_nw);
  }
}

/*
 * This function reads transfer function from tabulated file.
 * The table should be CAMB format (7th column corresponds to total components).
 */
void cosmology::read_transfer(const char *transfer_fname) {
  ifstream ifs;
  double dummy[7];
  string str;

  ifs.open(transfer_fname, ios::in);
  if (ifs.fail()) {
    cerr << "[ERROR] transfer file open error:" << transfer_fname << endl;
    exit(1);
  }

  while (getline(ifs, str)) {
    if (str[0] != '#') {
      sscanf(str.data(), "%lf %lf %lf %lf %lf %lf %lf", &dummy[0], &dummy[1],
             &dummy[2], &dummy[3], &dummy[4], &dummy[5], &dummy[6]);
      k.push_back(dummy[0]);
      Tk.push_back(dummy[6]);
    }
  }

  ifs.close();

  nk = k.size();
  for (int i = 0; i < nk; i++) {
    Tk[i] = sqr(k[i] * h) * Tk[i];
  }

  kmin_Tk = k[0];
  kmax_Tk = k[nk - 1];

  flag_transfer = true;

  return;
}

/*
 * loading transfer function at z = 0 from given arrays.
 * used from python module
 */
void cosmology::set_transfer(vector<double> k_set, vector<double> Tk_set) {
  if (k_set.size() != Tk_set.size()) {
    cerr << "[ERROR] The sizes of k and Tk should be the same." << endl;
    exit(1);
  }

  nk = k_set.size();

  k.resize(nk);
  Tk.resize(nk);

  for (int i = 0; i < nk; i++) {
    k[i] = k_set[i];
    Tk[i] = Tk_set[i];
  }

  for (int i = 0; i < nk; i++) {
    Tk[i] = sqr(k[i] * h) * Tk[i];
  }

  kmin_Tk = k[0];
  kmax_Tk = k[nk - 1];

  flag_transfer = true;

  return;
}

/*
 * This function sets fundamental spectra (linear power spectrum and
 * no-wiggle power spectrum) with the loaded transfer function.
 */
void cosmology::set_spectra(void) {
  double Delta_R, Delta_H, T_EH, Dplus, D0;
  double *P0, *Pno_wiggle0;

  if (!flag_transfer) {
    cerr << "[ERROR] Transfer function should be loaded first." << endl;
    exit(1);
  }

  Dplus = get_growth_factor(scale);
  D0 = get_growth_factor(1.0);

  P0 = new double[nk];
  Pno_wiggle0 = new double[nk];

  /* For an explicit formulation for the use of EH transfer function,
   * see Takada et al., PRD, 73, 083520, (2006).
   * We need to care about the transformation of the primordial curvature
   * fluctuation into the matter fluctuation.
   */
  for (int i = 0; i < nk; i++) {
    Delta_R = As * pow((k[i] * h) / k_pivot, ns - 1.0);
    Delta_H = 4.0 / 25.0 * As * pow((k[i] * h) / k_pivot, ns - 1.0);
    T_EH = nowiggle_EH_transfer(k[i]);
    Pno_wiggle0[i] = (2.0 * sqr(pi) / cub(k[i])) * Delta_H *
                     qua((k[i] * h) / (H0 / c_light)) / sqr(Omega_m) *
                     sqr(T_EH);
    Pno_wiggle0[i] *= sqr(D0);
    P0[i] = (2.0 * sqr(pi) / cub(k[i])) * Delta_R * sqr(Tk[i]);
  }

  P0min_Tk = P0[0];
  Pnw0min_Tk = Pno_wiggle0[0];

  acc = gsl_interp_accel_alloc();
  Pspl = gsl_spline_alloc(gsl_interp_cspline, nk);
  gsl_spline_init(Pspl, &k[0], P0, nk);

  acc_nw = gsl_interp_accel_alloc();
  Pspl_nw = gsl_spline_alloc(gsl_interp_cspline, nk);
  gsl_spline_init(Pspl_nw, &k[0], Pno_wiggle0, nk);

  delete[] P0;
  delete[] Pno_wiggle0;
  flag_set_spectra = true;

  return;
}

/*
 * This function sets fundamental spectra (linear power spectrum and
 * no-wiggle power spectrum).
 * In this function, no-wiggle spectrum is the input linear power spectrum.
 */
void cosmology::set_nw_spectra(void) {
  double Delta_H, T_EH, Dplus, D0;
  double *P0, *Pno_wiggle0;

  nk = 300;
  kmin_Tk = 1e-5;
  kmax_Tk = 1e3;

  k.resize(nk);
  Tk.resize(nk);

  for (int i = 0; i < nk; ++i) {
    k[i] = log(kmin_Tk) + (log(kmax_Tk) - log(kmin_Tk)) / ((double)nk) * i;
    k[i] = exp(k[i]);
    Tk[i] = 0.0;
  }

  P0 = new double[nk];
  Pno_wiggle0 = new double[nk];

  for (int i = 0; i < nk; i++) {
    Delta_H = 4.0 / 25.0 * As * pow((k[i] * h) / k_pivot, ns - 1.0);
    T_EH = nowiggle_EH_transfer(k[i]);
    Dplus = get_growth_factor(scale);
    D0 = get_growth_factor(1.0);
    Pno_wiggle0[i] = (2.0 * sqr(pi) / cub(k[i])) * Delta_H *
                     qua((k[i] * h) / (H0 / c_light)) / sqr(Omega_m) *
                     sqr(T_EH);
    Pno_wiggle0[i] *= sqr(D0);
    P0[i] = Pno_wiggle0[i];
  }

  P0min_Tk = P0[0];
  Pnw0min_Tk = Pno_wiggle0[0];

  acc = gsl_interp_accel_alloc();
  Pspl = gsl_spline_alloc(gsl_interp_cspline, nk);
  gsl_spline_init(Pspl, &k[0], P0, nk);

  acc_nw = gsl_interp_accel_alloc();
  Pspl_nw = gsl_spline_alloc(gsl_interp_cspline, nk);
  gsl_spline_init(Pspl_nw, &k[0], Pno_wiggle0, nk);

  delete[] P0;
  delete[] Pno_wiggle0;
  flag_set_spectra = true;

  return;
}

/*
 * This function gives growth factor at the given scale factor.
 * The growth factor is not normalized as D(a=1) = 1,
 * but it behaves as D(a) ~ a at high redshift like in EdS Universe.
 */
double cosmology::get_growth_factor(double a) {
  double res, E, ai, wi;

  res = 0.0;
  for (int i = 0; i < nint; ++i) {
    gsl_integration_glfixed_point(0.0, a, i, &ai, &wi, t);
    res +=
        wi * pow(Omega_m / ai + Omega_k + Omega_de * pow(ai, -1.0 - 3.0 * w_de),
                 -3.0 / 2.0);
  }

  E = sqrt(Omega_m / cub(a) + Omega_k / sqr(a) +
           Omega_de * pow(a, -3.0 * (1.0 + w_de)));
  res *= 5.0 / 2.0 * Omega_m * E;

  return res;
}

/*
 * Eisenstein and Hu transfer function fitting formula.
 * k should be in the unit of [h/Mpc].
 * Refs.: Eq [28-31] of Eisenstein and Hu, ApJ, 496, 605, (1998).
 */
double cosmology::nowiggle_EH_transfer(double k) {
  double T0, L0, C0, q, theta_CMB, alpha, s, gam_eff;

  s = 44.5 * log(9.83 / (Omega_m * h * h)) /
      sqrt(1.0 + 10.0 * pow(Omega_b * h * h, 3.0 / 4.0));
  s *= h; // Now, s is in the unit of [Mpc/h]
  alpha = 1.0 - 0.328 * log(431.0 * Omega_m * h * h) * Omega_b / Omega_m +
          0.38 * log(22.3 * Omega_m * h * h) * pow(Omega_b / Omega_m, 2.0);
  gam_eff =
      Omega_m * h * (alpha + (1.0 - alpha) / (1.0 + pow(0.43 * k * s, 4.0)));
  theta_CMB = T_CMB / 2.7;

  q = k * theta_CMB * theta_CMB / gam_eff;
  L0 = log(2.0 * e + 1.8 * q);
  C0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
  T0 = L0 / (L0 + C0 * q * q);

  return T0;
}

double cosmology::nowiggle_EH_power(double k) {
  double Delta_H, T_EH, Dplus, Pno_wiggle;

  Delta_H = 4.0 / 25.0 * As * pow((k * h) / k_pivot, ns - 1.0);
  T_EH = nowiggle_EH_transfer(k);
  Dplus = get_growth_factor(scale);

  Pno_wiggle = sqr(Dplus) * (2.0 * sqr(pi) / cub(k)) * Delta_H *
               qua((k * h) / (H0 / c_light)) / sqr(Omega_m) * sqr(T_EH);

  return Pno_wiggle;
}

/*
 * Computing the variance of density perturbation from the linear power
 * spectrum. R is in the unit of [Mpc/h].
 */
double cosmology::get_sigmaR(double R, double a) {
  double eta, ki, wi, res;

  if (!flag_set_spectra) {
    cerr << "[ERROR] Linear power spectrum should be set first." << endl;
    exit(1);
  }

  eta = get_eta(a);

  res = 0.0;
  for (int i = 0; i < nint; ++i) {
    gsl_integration_glfixed_point(log(kmin_Tk), log(kmax_Tk), i, &ki, &wi, t);
    ki = exp(ki);
    res += wi * cub(ki) *
           sqr(3.0 / cub(ki * R) * (sin(ki * R) - ki * R * cos(ki * R))) *
           exp(2.0 * eta) * P0(ki);
  }

  res = sqrt(res / (2.0 * sqr(pi)));

  return res;
}

/*
 * Computing conformal time.
 * eta = log(D), but normalized as eta0 = 0.
 */
double cosmology::get_eta(double a) {
  double res;

  res = log(get_growth_factor(a) / get_growth_factor(1.0));

  return res;
}

/*
 * Computing growth rate f = dlnD/dlna
 */
double cosmology::get_growth_rate(double a) {
  double dDda, E, dE, D, res;

  D = get_growth_factor(a);
  E = sqrt(Omega_m / cub(a) + Omega_k / sqr(a) +
           Omega_de * pow(a, -3.0 * (1.0 + w_de)));
  dE = 0.5 / E *
       (-3.0 * Omega_m / qua(a) - 2.0 * Omega_k / cub(a) -
        3.0 * (1.0 + w_de) * Omega_de * pow(a, -4.0 - 3.0 * w_de));
  dDda = D * dE / E + 5. / 2. * Omega_m / sqr(E) / cub(a);
  res = a / get_growth_factor(a) * dDda;

  return res;
}

/*
 * This function computes the comoving distance at the given scale factor.
 * Flat cosmology is assumed. The unit is [Mpc/h].
 */
double cosmology::get_comoving_distance(double a) {
  double ai, Ei, wi, res;

  res = 0.0;

  for (int i = 0; i < nint; ++i) {
    gsl_integration_glfixed_point(a, 1.0, i, &ai, &wi, t);
    Ei = sqrt(Omega_m / cub(ai) + Omega_k / sqr(ai) +
              Omega_de * pow(ai, -3.0 * (1.0 + w_de)));
    res += wi / (ai * ai * Ei);
  }

  res *= c_light / 100.0;

  return res;
}

/*
 * This function gives linear displacement dispersion at z = 0.
 * The unit is [Mpc/h]. No UV cutoff is applied.
 */
double cosmology::get_linear_displacement_dispersion(void) {
  double res, ki, wi;

  res = 0.0;
  for (int i = 0; i < nint; ++i) {
    gsl_integration_glfixed_point(log(kmin_Tk), log(kmax_Tk), i, &ki, &wi, t);
    ki = exp(ki);
    res += wi * ki * P0(ki);
  }
  res /= 6.0 * sqr(pi);
  res = sqrt(res);

  return res;
}

double cosmology::get_growth_factor(void) { return D0; }

double cosmology::get_eta(void) { return eta; }

double cosmology::get_growth_rate(void) { return f; }

double cosmology::get_comoving_distance(void) { return chi; }

double cosmology::get_sigma8(void) { return sigma8; }

double cosmology::get_Hubble(void) { return H; }

/* return linear power spectrum at z=0 */
double cosmology::P0(double k) {
  if (k <= kmin_Tk) {
    return pow(k / kmin_Tk, ns) * P0min_Tk;
  } else if (k > kmax_Tk)
    return 0.0;

  return gsl_spline_eval(Pspl, k, acc);
}

/* return linear power spectrum at the redshift of cosmology class */
double cosmology::Plin(double k) {
  if (k <= kmin_Tk) {
    return exp(2.0 * eta) * pow(k / kmin_Tk, ns) * P0min_Tk;
  } else if (k > kmax_Tk)
    return 0.0;

  return exp(2.0 * eta) * gsl_spline_eval(Pspl, k, acc);
}

/* return no-wiggle power spectrum at z=0 */
double cosmology::Pno_wiggle0(double k) {
  if (k <= kmin_Tk) {
    return pow(k / kmin_Tk, ns) * Pnw0min_Tk;
  } else if (k > kmax_Tk)
    return 0.0;

  return gsl_spline_eval(Pspl_nw, k, acc_nw);
}

/* return no-wiggle power spectrum at the redshift of the cosmology class */
double cosmology::Pno_wiggle(double k) {
  if (k <= kmin_Tk) {
    return exp(2.0 * eta) * pow(k / kmin_Tk, ns) * Pnw0min_Tk;
  } else if (k > kmax_Tk)
    return 0.0;

  return exp(2.0 * eta) * gsl_spline_eval(Pspl_nw, k, acc_nw);
}
