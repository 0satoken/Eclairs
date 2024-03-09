#include "bispectra.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <iostream>

bispectra::bispectra(params &params, cosmology &cosmo, spectra &spec){
  this->spec = &spec;

  eta = cosmo.get_eta();
  pi = 4.0 * atan(1.0);

  lambda = params.dparams["lambda_bispectrum"];

  /* integration setting */
  kmin = params.dparams["kmin"];
  kmax = params.dparams["kmax"];

  nq = params.iparams["direct_nq"];
  nmu = params.iparams["direct_nmu"];
  nphi = params.iparams["direct_nphi"];
  mumin = params.dparams["direct_mumin"];
  mumax = params.dparams["direct_mumax"];
  phimin = params.dparams["direct_phimin"];
  phimax = params.dparams["direct_phimax"];

  t_q = gsl_integration_glfixed_table_alloc(nq);
  t_mu = gsl_integration_glfixed_table_alloc(nmu);
  t_phi = gsl_integration_glfixed_table_alloc(nphi);

  q = new double[nq];
  mu = new double[nmu];
  phi = new double[nphi];
  wq = new double[nq];
  wmu = new double[nmu];
  wphi = new double[nphi];

  for(int iq=0;iq<nq;++iq){
    gsl_integration_glfixed_point(log(kmin), log(kmax), iq, &q[iq], &wq[iq], t_q);
    q[iq] = exp(q[iq]);
  }

  for(int imu=0;imu<nmu;++imu){
    gsl_integration_glfixed_point(mumin, mumax, imu, &mu[imu], &wmu[imu], t_mu);
  }

  for(int iphi=0;iphi<nphi;++iphi){
    gsl_integration_glfixed_point(phimin, phimax, iphi, &phi[iphi], &wphi[iphi], t_phi);
  }

  /* setting spline functions of sigmad and power spectra at 1-loop for B-term */
  nk_spl = params.iparams["nk_spl"];
  set_sigmad2_spline();

  /* flags */
  flag_SPT = params.bparams["direct_SPT"];
}

bispectra::~bispectra(){
  gsl_integration_glfixed_table_free(t_q);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_integration_glfixed_table_free(t_phi);
  delete[] q;
  delete[] mu;
  delete[] phi;
  delete[] wq;
  delete[] wmu;
  delete[] wphi;
  gsl_interp_accel_free(acc_sigmad2);
  gsl_spline_free(spl_sigmad2);
}

void bispectra::set_sigmad2_spline(void){
  double *logk_table, *sigmad2_table;


  logk_table = new double[nk_spl];
  sigmad2_table = new double[nk_spl];

  acc_sigmad2 = gsl_interp_accel_alloc();

  for (int i=0;i<nk_spl;++i){
    logk_table[i] = (log(kmax) - log(kmin)) / (nk_spl - 1.0) * i + log(kmin);
    sigmad2_table[i] = spec->get_sigmad2(exp(logk_table[i]), lambda);
  }

  sigmad2_max = sigmad2_table[nk_spl-1];

  spl_sigmad2 = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  gsl_spline_init(spl_sigmad2, logk_table, sigmad2_table, nk_spl);

  delete[] logk_table;
  delete[] sigmad2_table;

  return;
}

double bispectra::sigmad2_spl(double k) {
  double logk;

  logk = log(k);
  if (logk < log(kmin)) {
    return 0.0;
  } else if (logk > log(kmax)) {
    return sigmad2_max;
  } else {
    return gsl_spline_eval(spl_sigmad2, logk, acc_sigmad2);
  }
}

/* Bispectrum (1-loop) with IR-safe integral */
double bispectra::Bispec_1loop(Type a, Type b, Type c, double k1, double k2, double k3){
  double res, D, Bk222, Bk321, mu12, k12, k23, k31;
  double Bk211, G1reg_k1, G1reg_k2, G1reg_k3, G2reg_k1_k2, G2reg_k2_k3, G2reg_k3_k1;
  double F2_k1_k2, F2_k2_k3, F2_k3_k1;
  double G1_1l_k1, G1_1l_k2, G1_1l_k3, G2_1l_k1_k2, G2_1l_k2_k3, G2_1l_k3_k1;
  double sigmad2_k1, sigmad2_k2, sigmad2_k3, sigmad2_k12, sigmad2_k23, sigmad2_k31;
  double integ_Bk222, integ_Bk321, integ_mu_Bk222, integ_mu_Bk321, qi;
  double p1, p2, p3, r1, r2, r3;
  double Pk1, Pk2, Pk3, Pkq, Pkp1, Pkp2, Pkp3, Pkr1, Pkr2, Pkr3;
  Vector kk1, kk2, kk3, qq;
  Vector pp1, pp2, pp3, rr1, rr2, rr3;


  /* triangular condition is not satisfied */
  if(k1 > k2 + k3 || k2 > k3 + k1 || k3 > k1 + k2)
    return 0.0;

  D = exp(eta);
  mu12 = -(sqr(k1) + sqr(k2) - sqr(k3)) / (2.0 * k1 * k2);

  kk1.x = 0.0, kk1.y = 0.0, kk1.z = k1;
  kk2.x = 0.0, kk2.y = k2 * sqrt(fabs(1.0 - sqr(mu12))), kk2.z = k2 * mu12;
  kk3 = -kk1 - kk2;

  Pk1 = spec->P0(k1);
  Pk2 = spec->P0(k2);
  Pk3 = spec->P0(k3);

  k12 = sqrt((kk1 + kk2) * (kk1 + kk2));
  k23 = sqrt((kk2 + kk3) * (kk2 + kk3));
  k31 = sqrt((kk3 + kk1) * (kk3 + kk1));

  sigmad2_k1 = sigmad2_spl(k1) * sqr(D);
  sigmad2_k2 = sigmad2_spl(k2) * sqr(D);
  sigmad2_k3 = sigmad2_spl(k3) * sqr(D);
  sigmad2_k12 = sigmad2_spl(k12) * sqr(D);
  sigmad2_k23 = sigmad2_spl(k23) * sqr(D);
  sigmad2_k31 = sigmad2_spl(k31) * sqr(D);

  Bk222 = 0.0;
  Bk321 = 0.0;
  for (int iq = 0; iq < nq; iq++) {
    qi = q[iq];
    Pkq = spec->P0(qi);
    integ_Bk222 = 0.0;
    integ_Bk321 = 0.0;
    for (int imu = 0; imu < nmu; imu++) {
      integ_mu_Bk222 = 0.0;
      integ_mu_Bk321 = 0.0;
      for (int iphi = 0; iphi < nphi; iphi++) {
        qq.x = qi * sqrt(1.0 - sqr(mu[imu])) * cos(phi[iphi]);
        qq.y = qi * sqrt(1.0 - sqr(mu[imu])) * sin(phi[iphi]);
        qq.z = qi * mu[imu];

        pp1 = kk1 - qq;
        pp2 = kk2 - qq;
        pp3 = kk3 - qq;
        rr1 = kk1 + qq;
        rr2 = kk2 + qq;
        rr3 = kk3 + qq;

        p1 = sqrt(pp1 * pp1);
        p2 = sqrt(pp2 * pp2);
        p3 = sqrt(pp3 * pp3);
        r1 = sqrt(rr1 * rr1);
        r2 = sqrt(rr2 * rr2);
        r3 = sqrt(rr3 * rr3);

        Pkp1 = spec->P0(p1);
        Pkp2 = spec->P0(p2);
        Pkp3 = spec->P0(p3);
        Pkr1 = spec->P0(r1);
        Pkr2 = spec->P0(r2);
        Pkr3 = spec->P0(r3);

        if (flag_SPT) {
          // IR-safe integrand for Bk222 with SPT
          if (p1 > qi && r2 > qi)
            integ_mu_Bk222 += wphi[iphi] * F2_sym(a, pp1, qq) *
                              F2_sym(b, rr2, -qq) * F2_sym(c, -rr2, -pp1) *
                              Pkp1 * Pkq * Pkr2;

          if (r1 > qi && p2 > qi)
            integ_mu_Bk222 += wphi[iphi] * F2_sym(a, rr1, -qq) *
                              F2_sym(b, pp2, qq) * F2_sym(c, -pp2, -rr1) *
                              Pkr1 * Pkq * Pkp2;

          if (p3 > qi && r2 > qi)
            integ_mu_Bk222 += wphi[iphi] * F2_sym(c, pp3, qq) *
                              F2_sym(b, rr2, -qq) * F2_sym(a, -rr2, -pp3) *
                              Pkp3 * Pkq * Pkr2;

          if (r3 > qi && p2 > qi)
            integ_mu_Bk222 += wphi[iphi] * F2_sym(c, rr3, -qq) *
                              F2_sym(b, pp2, qq) * F2_sym(a, -pp2, -rr3) *
                              Pkr3 * Pkq * Pkp2;

          if (p1 > qi && r3 > qi)
            integ_mu_Bk222 += wphi[iphi] * F2_sym(a, pp1, qq) *
                              F2_sym(c, rr3, -qq) * F2_sym(b, -rr3, -pp1) *
                              Pkp1 * Pkq * Pkr3;

          if (r1 > qi && p3 > qi)
            integ_mu_Bk222 += wphi[iphi] * F2_sym(a, rr1, -qq) *
                              F2_sym(c, pp3, qq) * F2_sym(b, -pp3, -rr1) *
                              Pkr1 * Pkq * Pkp3;

          // IR-safe integrand for Bk321 with SPT
          if (p2 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] * (F3_sym(a, -kk3, -pp2, -qq) * F2_sym(b, pp2, qq) *
                                  Pkp2 * Pkq * Pk3 +
                              F3_sym(c, -kk1, -pp2, -qq) * F2_sym(b, pp2, qq) *
                                  Pkp2 * Pkq * Pk1);

          if (r2 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] * (F3_sym(a, -kk3, -rr2, qq) * F2_sym(b, rr2, -qq) *
                                  Pkr2 * Pkq * Pk3 +
                              F3_sym(c, -kk1, -rr2, qq) * F2_sym(b, rr2, -qq) *
                                  Pkr2 * Pkq * Pk1);

          if (p3 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] * (F3_sym(a, -kk2, -pp3, -qq) * F2_sym(c, pp3, qq) *
                                  Pkp3 * Pkq * Pk2 +
                              F3_sym(b, -kk1, -pp3, -qq) * F2_sym(c, pp3, qq) *
                                  Pkp3 * Pkq * Pk1);

          if (r3 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] * (F3_sym(a, -kk2, -rr3, qq) * F2_sym(c, rr3, -qq) *
                                  Pkr3 * Pkq * Pk2 +
                              F3_sym(b, -kk1, -rr3, qq) * F2_sym(c, rr3, -qq) *
                                  Pkr3 * Pkq * Pk1);

          if (p1 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] * (F3_sym(b, -kk3, -pp1, -qq) * F2_sym(a, pp1, qq) *
                                  Pkp1 * Pkq * Pk3 +
                              F3_sym(c, -kk2, -pp1, -qq) * F2_sym(a, pp1, qq) *
                                  Pkp1 * Pkq * Pk2);

          if (r1 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] * (F3_sym(b, -kk3, -rr1, qq) * F2_sym(a, rr1, -qq) *
                                  Pkr1 * Pkq * Pk3 +
                              F3_sym(c, -kk2, -rr1, qq) * F2_sym(a, rr1, -qq) *
                                  Pkr1 * Pkq * Pk2);
        } else {
          // IR-safe integrand for Bk222
          if (p1 > qi && r2 > qi)
            integ_mu_Bk222 +=
                wphi[iphi] * F2_sym(a, pp1, qq) * F2_sym(b, rr2, -qq) *
                F2_sym(c, -rr2, -pp1) * Pkp1 * Pkq * Pkr2 *
                exp(-0.5 * (sqr(k1) * sigmad2_k1 + sqr(k2) * sigmad2_k2 +
                            sqr(k12) * sigmad2_k12));

          if (r1 > qi && p2 > qi)
            integ_mu_Bk222 +=
                wphi[iphi] * F2_sym(a, rr1, -qq) * F2_sym(b, pp2, qq) *
                F2_sym(c, -pp2, -rr1) * Pkr1 * Pkq * Pkp2 *
                exp(-0.5 * (sqr(k1) * sigmad2_k1 + sqr(k2) * sigmad2_k2 +
                            sqr(k12) * sigmad2_k12));

          if (p3 > qi && r2 > qi)
            integ_mu_Bk222 +=
                wphi[iphi] * F2_sym(c, pp3, qq) * F2_sym(b, rr2, -qq) *
                F2_sym(a, -rr2, -pp3) * Pkp3 * Pkq * Pkr2 *
                exp(-0.5 * (sqr(k2) * sigmad2_k2 + sqr(k3) * sigmad2_k3 +
                            sqr(k23) * sigmad2_k23));

          if (r3 > qi && p2 > qi)
            integ_mu_Bk222 +=
                wphi[iphi] * F2_sym(c, rr3, -qq) * F2_sym(b, pp2, qq) *
                F2_sym(a, -pp2, -rr3) * Pkr3 * Pkq * Pkp2 *
                exp(-0.5 * (sqr(k2) * sigmad2_k2 + sqr(k3) * sigmad2_k3 +
                            sqr(k23) * sigmad2_k23));

          if (p1 > qi && r3 > qi)
            integ_mu_Bk222 +=
                wphi[iphi] * F2_sym(a, pp1, qq) * F2_sym(c, rr3, -qq) *
                F2_sym(b, -rr3, -pp1) * Pkp1 * Pkq * Pkr3 *
                exp(-0.5 * (sqr(k3) * sigmad2_k3 + sqr(k1) * sigmad2_k1 +
                            sqr(k31) * sigmad2_k31));

          if (r1 > qi && p3 > qi)
            integ_mu_Bk222 +=
                wphi[iphi] * F2_sym(a, rr1, -qq) * F2_sym(c, pp3, qq) *
                F2_sym(b, -pp3, -rr1) * Pkr1 * Pkq * Pkp3 *
                exp(-0.5 * (sqr(k3) * sigmad2_k3 + sqr(k1) * sigmad2_k1 +
                            sqr(k31) * sigmad2_k31));

          // IR-safe integrand for Bk321
          if (p2 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] *
                (F3_sym(a, -kk3, -pp2, -qq) * F2_sym(b, pp2, qq) * Pkp2 * Pkq *
                     Pk3 *
                     exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                                 sqr(k3) * sigmad2_k3)) +
                 F3_sym(c, -kk1, -pp2, -qq) * F2_sym(b, pp2, qq) * Pkp2 * Pkq *
                     Pk1 *
                     exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                                 sqr(k2) * sigmad2_k2)));

          if (r2 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] *
                (F3_sym(a, -kk3, -rr2, qq) * F2_sym(b, rr2, -qq) * Pkr2 * Pkq *
                     Pk3 *
                     exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                                 sqr(k3) * sigmad2_k3)) +
                 F3_sym(c, -kk1, -rr2, qq) * F2_sym(b, rr2, -qq) * Pkr2 * Pkq *
                     Pk1 *
                     exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                                 sqr(k2) * sigmad2_k2)));

          if (p3 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] *
                (F3_sym(a, -kk2, -pp3, -qq) * F2_sym(c, pp3, qq) * Pkp3 * Pkq *
                     Pk2 *
                     exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                                 sqr(k3) * sigmad2_k3)) +
                 F3_sym(b, -kk1, -pp3, -qq) * F2_sym(c, pp3, qq) * Pkp3 * Pkq *
                     Pk1 *
                     exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                                 sqr(k1) * sigmad2_k1)));

          if (r3 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] *
                (F3_sym(a, -kk2, -rr3, qq) * F2_sym(c, rr3, -qq) * Pkr3 * Pkq *
                     Pk2 *
                     exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                                 sqr(k3) * sigmad2_k3)) +
                 F3_sym(b, -kk1, -rr3, qq) * F2_sym(c, rr3, -qq) * Pkr3 * Pkq *
                     Pk1 *
                     exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                                 sqr(k1) * sigmad2_k1)));

          if (p1 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] *
                (F3_sym(b, -kk3, -pp1, -qq) * F2_sym(a, pp1, qq) * Pkp1 * Pkq *
                     Pk3 *
                     exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                                 sqr(k1) * sigmad2_k1)) +
                 F3_sym(c, -kk2, -pp1, -qq) * F2_sym(a, pp1, qq) * Pkp1 * Pkq *
                     Pk2 *
                     exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                                 sqr(k2) * sigmad2_k2)));

          if (r1 > qi)
            integ_mu_Bk321 +=
                wphi[iphi] *
                (F3_sym(b, -kk3, -rr1, qq) * F2_sym(a, rr1, -qq) * Pkr1 * Pkq *
                     Pk3 *
                     exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                                 sqr(k1) * sigmad2_k1)) +
                 F3_sym(c, -kk2, -rr1, qq) * F2_sym(a, rr1, -qq) * Pkr1 * Pkq *
                     Pk2 *
                     exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                                 sqr(k2) * sigmad2_k2)));
        }
      }
      integ_mu_Bk222 /= 2.0;

      integ_Bk222 += wmu[imu] * integ_mu_Bk222;
      integ_Bk321 += wmu[imu] * integ_mu_Bk321;
    }
    Bk222 += integ_Bk222 * wq[iq] * cub(qi);
    Bk321 += integ_Bk321 * wq[iq] * cub(qi);
  }

  Bk222 *= 8.0 / cub(2.0 * pi);
  Bk321 *= 6.0 / cub(2.0 * pi);

  F2_k1_k2 = F2_sym(c, kk1, kk2);
  F2_k2_k3 = F2_sym(a, kk2, kk3);
  F2_k3_k1 = F2_sym(b, kk3, kk1);

  G1_1l_k1 = spec->Gamma1_1loop(a, k1);
  G1_1l_k2 = spec->Gamma1_1loop(b, k2);
  G1_1l_k3 = spec->Gamma1_1loop(c, k3);

  G2_1l_k1_k2 = spec->Gamma2_1loop(c, k1, k2, k3);
  G2_1l_k2_k3 = spec->Gamma2_1loop(a, k2, k3, k1);
  G2_1l_k3_k1 = spec->Gamma2_1loop(b, k3, k1, k2);

  if (flag_SPT) {
    Bk211 = 2.0 * ((F2_k2_k3 + sqr(D) * G2_1l_k2_k3 +
                    sqr(D) * F2_k2_k3 * (G1_1l_k2 + G1_1l_k3)) *
                       Pk2 * Pk3 +
                   (F2_k3_k1 + sqr(D) * G2_1l_k3_k1 +
                    sqr(D) * F2_k3_k1 * (G1_1l_k3 + G1_1l_k1)) *
                       Pk3 * Pk1 +
                   (F2_k1_k2 + sqr(D) * G2_1l_k1_k2 +
                    sqr(D) * F2_k1_k2 * (G1_1l_k1 + G1_1l_k2)) *
                       Pk1 * Pk2);
  } else {
    G2reg_k1_k2 = (F2_k1_k2 * (1.0 + 0.5 * sqr(k12) * sigmad2_k12) +
                   G2_1l_k1_k2 * sqr(D)) *
                  exp(-0.5 * sqr(k12) * sigmad2_k12);
    G2reg_k2_k3 = (F2_k2_k3 * (1.0 + 0.5 * sqr(k23) * sigmad2_k23) +
                   G2_1l_k2_k3 * sqr(D)) *
                  exp(-0.5 * sqr(k23) * sigmad2_k23);
    G2reg_k3_k1 = (F2_k3_k1 * (1.0 + 0.5 * sqr(k31) * sigmad2_k31) +
                   G2_1l_k3_k1 * sqr(D)) *
                  exp(-0.5 * sqr(k31) * sigmad2_k31);

    G1reg_k1 = (1.0 + 0.5 * sqr(k1) * sigmad2_k1 + G1_1l_k1 * sqr(D)) *
               exp(-0.5 * sqr(k1) * sigmad2_k1);
    G1reg_k2 = (1.0 + 0.5 * sqr(k2) * sigmad2_k2 + G1_1l_k2 * sqr(D)) *
               exp(-0.5 * sqr(k2) * sigmad2_k2);
    G1reg_k3 = (1.0 + 0.5 * sqr(k3) * sigmad2_k3 + G1_1l_k3 * sqr(D)) *
               exp(-0.5 * sqr(k3) * sigmad2_k3);

    Bk211 = 2.0 * (G2reg_k2_k3 * G1reg_k2 * G1reg_k3 * Pk2 * Pk3 +
                   G2reg_k3_k1 * G1reg_k3 * G1reg_k1 * Pk3 * Pk1 +
                   G2reg_k1_k2 * G1reg_k1 * G1reg_k2 * Pk1 * Pk2);
  }

  res = qua(D) * Bk211 + sqr(D) * qua(D) * (Bk222 + Bk321);

  return res;
}

/* Bispectrum (tree level) */
double bispectra::Bispec_tree(Type a, Type b, Type c, double k1, double k2,
                              double k3) {
  double res, D, mu12, k12, k23, k31;
  double sigmad2_k1, sigmad2_k2, sigmad2_k3, sigmad2_k12, sigmad2_k23,
      sigmad2_k31;
  double Pk1, Pk2, Pk3;
  Vector kk1, kk2, kk3;

  if (k1 > k2 + k3 || k2 > k3 + k1 || k3 > k1 + k2)
    return 0.0;

  D = exp(eta);

  mu12 = -(sqr(k1) + sqr(k2) - sqr(k3)) / (2.0 * k1 * k2);
  kk1.x = 0.0, kk1.y = 0.0, kk1.z = k1;
  kk2.x = 0.0, kk2.y = k2 * sqrt(fabs(1.0 - sqr(mu12))), kk2.z = k2 * mu12;
  kk3 = -kk1 - kk2;

  Pk1 = spec->P0(k1);
  Pk2 = spec->P0(k2);
  Pk3 = spec->P0(k3);

  k12 = sqrt((kk1 + kk2) * (kk1 + kk2));
  k23 = sqrt((kk2 + kk3) * (kk2 + kk3));
  k31 = sqrt((kk3 + kk1) * (kk3 + kk1));

  sigmad2_k1 = sigmad2_spl(k1) * sqr(D);
  sigmad2_k2 = sigmad2_spl(k2) * sqr(D);
  sigmad2_k3 = sigmad2_spl(k3) * sqr(D);
  sigmad2_k12 = sigmad2_spl(k12) * sqr(D);
  sigmad2_k23 = sigmad2_spl(k23) * sqr(D);
  sigmad2_k31 = sigmad2_spl(k31) * sqr(D);

  if (flag_SPT) {
    res = 2.0 *
          (F2_sym(a, kk2, kk3) * Pk2 * Pk3 + F2_sym(b, kk3, kk1) * Pk3 * Pk1 +
           F2_sym(c, kk1, kk2) * Pk1 * Pk2);
  } else {
    res = 2.0 * (F2_sym(a, kk2, kk3) * Pk2 * Pk3 *
                     exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                                 sqr(k3) * sigmad2_k3)) +
                 F2_sym(b, kk3, kk1) * Pk3 * Pk1 *
                     exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                                 sqr(k1) * sigmad2_k1)) +
                 F2_sym(c, kk1, kk2) * Pk1 * Pk2 *
                     exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                                 sqr(k2) * sigmad2_k2)));
  }

  res = qua(D) * res;

  return res;
}
