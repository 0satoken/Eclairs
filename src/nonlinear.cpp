#include "nonlinear.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "vector.hpp"
#include <algorithm>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <iostream>
#include <string>

using namespace std;

nonlinear::nonlinear(params &params, cosmology &cosmo) {
  this->cosmo = &cosmo;
  pi = 4.0 * atan(1.0);
  kmin = params.dparams["kmin"];
  kmax = params.dparams["kmax"];
  nint = params.iparams["nint"]; // number of steps for generic integration
  iter_max = 1000;

  /* cosmological parameters */
  w_de = params.dparams["w_de"];
  h = params.dparams["H0"] / 100.0;
  Omega_m = params.dparams["Omega_m"];
  Omega_de = 1.0 - Omega_m;
  fnu = params.dparams["m_nu"] / (93.14 * h * h); // neutrino fraction
  z = params.dparams["z"];

  eta = log(cosmo.get_growth_factor(1.0 / (1.0 + z)) /
            cosmo.get_growth_factor(1.0));
  sigma8 = cosmo.get_sigmaR(8.0, 1.0 / (1.0 + z));

  find_ksigma();
  set_omega();
}

nonlinear::~nonlinear() {
}

double nonlinear::Plinz(double k) {
  return exp(2.0 * eta) * cosmo->P0(k);
}

void nonlinear::find_ksigma(void) {
  double res, res_deriv, res_deriv2, qi, R;
  double logR1, logR2, Rmid;
  double logq[nint], q[nint], w[nint];
  int cnt;
  gsl_integration_glfixed_table *t;

  t = gsl_integration_glfixed_table_alloc(nint);
  for (int i = 0; i < nint; ++i) {
    gsl_integration_glfixed_point(log(kmin), log(kmax), i, &logq[i], &w[i], t);
    q[i] = exp(logq[i]);
  }

  // Find ksigma by binary search
  logR1 = -2.0;
  logR2 = 3.5;
  cnt = 0;
  do {
    Rmid = 0.5 * (logR1 + logR2);
    Rmid = pow(10.0, Rmid);
    res = 0.0;
    for (int i = 0; i < nint; ++i) {
      qi = q[i];
      res += w[i] * cub(qi) * Plinz(qi) / (2.0 * pi * pi) *
             exp(-sqr(qi) * sqr(Rmid));
    }
    if (res - 1.0 > 0.0)
      logR1 = log10(Rmid);
    else if (res - 1.0 < 0.0)
      logR2 = log10(Rmid);
    cnt++;
    if (logR2 < -1.999) {
      Rmid = 0.01;
      break;
    }
  } while (fabs(res - 1.0) > 1e-5 && cnt < iter_max);
  R = Rmid;

  /*
    //Find ksigma by Newton-Raphson method //
    R = 0.1;
    cnt = 0;
    do{
      res = 0.0;
      res_deriv = 0.0;
      for(int i=0;i<nint;++i){
        qi = q[i];
        res += w[i]*cub(qi)*Plinz(qi)/(2.0*pi*pi)*exp(-sqr(qi)*sqr(R));
        res_deriv +=
    w[i]*cub(qi)*Plinz(qi)/(2.0*pi*pi)*exp(-sqr(qi)*sqr(R))*(-2.0*R*sqr(qi));
      }
      R = R - (res-1.0)/res_deriv;
      cnt++;
    }while(fabs(res-1.0) > 1e-5 && cnt < iter_max);
  */

  if (cnt == iter_max) {
    cerr << "Finding ksigma does not converge!" << endl;
    exit(1);
  }

  res = 0.0;
  res_deriv = 0.0;
  res_deriv2 = 0.0;
  for (int i = 0; i < nint; ++i) {
    qi = q[i];
    res +=
        w[i] * cub(qi) * Plinz(qi) / (2.0 * pi * pi) * exp(-sqr(qi) * sqr(R));
    res_deriv += w[i] * cub(qi) * Plinz(qi) / (2.0 * pi * pi) *
                 exp(-sqr(qi) * sqr(R)) * (-2.0 * R * sqr(qi));
    res_deriv2 += w[i] * cub(qi) * Plinz(qi) / (2.0 * pi * pi) *
                  exp(-sqr(qi) * sqr(R)) *
                  (-2.0 * sqr(qi) + 4.0 * sqr(R) * qua(qi));
  }

  rn = -3.0 - R / res * res_deriv;
  rncur = -R * (res_deriv / res - R / sqr(res) * sqr(res_deriv) +
                R / res * res_deriv2);
  ksigma = 1.0 / R;

  gsl_integration_glfixed_table_free(t);

  return;
}

void nonlinear::set_omega(void) {
  double a, E;

  a = 1.0 / (1.0 + z);
  E = sqrt(Omega_m * pow(a, -3.0) + Omega_de * pow(a, -3.0 * (1.0 + w_de)));

  om_m = Omega_m / cub(a) / sqr(E);
  om_v = Omega_de * pow(a, -3.0 * (1.0 + w_de)) / sqr(E);

  return;
}

double nonlinear::Pk_halofit(double k) {
  double DeltaL, DeltaQ, DeltaH, DeltaHp, y;
  double f1a, f2a, f3a, f1b, f2b, f3b, frac;
  double an, bn, cn, alpha, beta, gamma, xmu, xnu;
  double f1, f2, f3;

  if (k < kmin)
    return Plinz(k);
  else if (k > kmax)
    return 0.0;

  gamma = 0.1971 - 0.0843 * rn + 0.8460 * rncur;
  an = 1.5222 + 2.8553 * rn + 2.3706 * sqr(rn) + 0.9903 * cub(rn) +
       0.2250 * qua(rn) - 0.6038 * rncur + 0.1749 * om_v * (1.0 + w_de);
  an = pow(10.0, an);
  bn = -0.5642 + 0.5864 * rn + 0.5716 * sqr(rn) - 1.5474 * rncur +
       0.2279 * om_v * (1.0 + w_de);
  bn = pow(10.0, bn);
  cn = 0.3698 + 2.0404 * rn + 0.8161 * sqr(rn) + 0.5869 * rncur;
  cn = pow(10.0, cn);
  xmu = 0.0;
  xnu = pow(10, 5.2105 + 3.6902 * rn);
  alpha = fabs(6.0835 + 1.3373 * rn - 0.1959 * sqr(rn) - 5.5274 * rncur);
  beta = 2.0379 - 0.7354 * rn + 0.3157 * sqr(rn) + 1.2490 * cub(rn) +
         0.3980 * qua(rn) - 0.1682 * rncur + fnu * (1.081 + 0.395 * sqr(rn));

  if (fabs(1.0 - om_m) > 0.01) {
    f1a = pow(om_m, -0.0732);
    f2a = pow(om_m, -0.1423);
    f3a = pow(om_m, 0.0725);
    f1b = pow(om_m, -0.0307);
    f2b = pow(om_m, -0.0585);
    f3b = pow(om_m, 0.0743);
    frac = om_v / (1.0 - om_m);
    f1 = frac * f1b + (1 - frac) * f1a;
    f2 = frac * f2b + (1 - frac) * f2a;
    f3 = frac * f3b + (1 - frac) * f3a;
  } else {
    f1 = 1.0;
    f2 = 1.0;
    f3 = 1.0;
  }

  y = k / ksigma;
  DeltaL = cub(k) / (2.0 * sqr(pi)) * Plinz(k);

  DeltaQ =
      DeltaL *
      (pow(1.0 + DeltaL * (1.0 + fnu * 47.48 * sqr(k) / (1.0 + 1.5 * sqr(k))),
           beta) /
       (1.0 + DeltaL * alpha)) *
      exp(-y / 4.0 - sqr(y) / 8.0);

  DeltaHp = an * pow(y, 3.0 * f1) /
            (1.0 + bn * pow(y, f2) + pow(cn * f3 * y, 3.0 - gamma));
  DeltaH = DeltaHp / (1.0 + xmu / y + xnu / sqr(y)) * (1.0 + fnu * 0.977);

  return (2.0 * sqr(pi)) / cub(k) * (DeltaQ + DeltaH);
}

double nonlinear::Bk_halofit(double k1, double k2, double k3) {
  double an, bn, cn, alpha, beta, gamma, r1, r2;
  double dn, en, fn, gn, hn, mn, nn, mun, nun, pn;
  double q1, q2, q3;
  double Ik1, Ik2, Ik3, PEk1, PEk2, PEk3;
  double B1h, B3h, res;
  double ks[3];

  if (k1 + k2 < k3 || k2 + k3 < k1 || k3 + k1 < k2) {
    return 0.0;
  }

  ks[0] = k1;
  ks[1] = k2;
  ks[2] = k3;

  sort(ks, ks + 3);

  r1 = ks[0] / ks[2];
  r2 = (ks[1] + ks[0] - ks[2]) / ks[2];

  q1 = k1 / ksigma;
  q2 = k2 / ksigma;
  q3 = k3 / ksigma;

  alpha = min(-4.348 - 3.006 * rn - 0.5745 * sqr(rn) +
                  pow(10.0, -0.9 + 0.2 * rn) * sqr(r2),
              log10(1.0 - 2.0 / 3.0 * rn));
  alpha = pow(10.0, alpha);
  beta = -1.731 - 2.845 * rn - 1.4995 * sqr(rn) - 0.2811 * cub(rn) + 0.007 * r2;
  beta = pow(10.0, beta);
  gamma = 0.182 + 0.570 * rn;
  gamma = pow(10.0, gamma);
  an = -2.167 - 2.944 * log10(sigma8) - 1.106 * sqr(log10(sigma8)) -
       2.865 * cub(log10(sigma8)) - 0.310 * pow(r1, gamma);
  an = pow(10.0, an);
  bn = -3.428 - 2.681 * log10(sigma8) + 1.624 * sqr(log10(sigma8)) -
       0.095 * cub(log10(sigma8));
  bn = pow(10.0, bn);
  cn = 0.159 - 1.107 * rn;
  cn = pow(10.0, cn);

  dn = -0.483 + 0.892 * log10(sigma8) - 0.086 * om_m;
  dn = pow(10.0, dn);
  en = -0.632 + 0.646 * rn;
  en = pow(10.0, en);
  fn = -10.533 - 16.838 * rn - 9.3048 * sqr(rn) - 1.8263 * cub(rn);
  fn = pow(10.0, fn);
  gn = 2.787 + 2.405 * rn + 0.4577 * sqr(rn);
  gn = pow(10.0, gn);
  hn = -1.188 - 0.394 * rn;
  hn = pow(10.0, hn);
  mn = -2.605 - 2.434 * log10(sigma8) + 5.710 * sqr(log10(sigma8));
  mn = pow(10.0, mn);
  nn = -4.468 - 3.080 * log10(sigma8) + 1.035 * sqr(log10(sigma8));
  nn = pow(10.0, nn);
  mun = 15.312 + 22.977 * rn + 10.9579 * sqr(rn) + 1.6586 * cub(rn);
  mun = pow(10.0, mun);
  nun = 1.347 + 1.246 * rn + 0.4525 * sqr(rn);
  nun = pow(10.0, nun);
  pn = 0.071 - 0.433 * rn;
  pn = pow(10.0, pn);

  B1h = 1.0 /
        ((an * pow(q1, alpha) + bn * pow(q1, beta)) * (1.0 + 1.0 / (cn * q1)) *
         (an * pow(q2, alpha) + bn * pow(q2, beta)) * (1.0 + 1.0 / (cn * q2)) *
         (an * pow(q3, alpha) + bn * pow(q3, beta)) * (1.0 + 1.0 / (cn * q3)));

  Ik1 = 1.0 / (1.0 + en * q1);
  Ik2 = 1.0 / (1.0 + en * q2);
  Ik3 = 1.0 / (1.0 + en * q3);

  PEk1 = (1.0 + fn * sqr(q1)) / (1.0 + gn * q1 + hn * sqr(q1)) * Plinz(k1) +
         1.0 / ((mn * pow(q1, mun) + nn * pow(q1, nun)) *
                (1.0 + 1.0 / cub(pn * q1)));
  PEk2 = (1.0 + fn * sqr(q2)) / (1.0 + gn * q2 + hn * sqr(q2)) * Plinz(k2) +
         1.0 / ((mn * pow(q2, mun) + nn * pow(q2, nun)) *
                (1.0 + 1.0 / cub(pn * q2)));
  PEk3 = (1.0 + fn * sqr(q3)) / (1.0 + gn * q3 + hn * sqr(q3)) * Plinz(k3) +
         1.0 / ((mn * pow(q3, mun) + nn * pow(q3, nun)) *
                (1.0 + 1.0 / cub(pn * q3)));

  B3h = 2.0 * (F2(k1, k2, k3) + dn * q3) * Ik1 * Ik2 * Ik3 * PEk1 * PEk2 +
        2.0 * (F2(k2, k3, k1) + dn * q1) * Ik2 * Ik3 * Ik1 * PEk2 * PEk3 +
        2.0 * (F2(k3, k1, k2) + dn * q2) * Ik3 * Ik1 * Ik2 * PEk3 * PEk1;

  res = B1h + B3h;

  return res;
}

double nonlinear::F2(double k1, double k2, double k3) {
  double res, k1k2;

  k1k2 = (sqr(k3) - sqr(k1) - sqr(k2)) / 2.0;
  res = 5.0 / 7.0 + 0.5 * k1k2 / (k1 * k2) * (k1 / k2 + k2 / k1) +
        2.0 / 7.0 * sqr(k1k2 / (k1 * k2));

  return res;
}

void nonlinear::change_redshift(double z) {
  this->z = z;
  this->eta = log(cosmo->get_growth_factor(1.0 / (1.0 + z)) /
                  cosmo->get_growth_factor(1.0));
  this->sigma8 = cosmo->get_sigmaR(8.0, 1.0 / (1.0 + z));

  find_ksigma();
  set_omega();
}
