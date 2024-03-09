#include "fast_kernels_bispec.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_spline.h>
#include <iostream>

fast_kernels_bispec::fast_kernels_bispec(params &params, spectra &spec,
                                         int myrank, int numprocs) {
  this->para = &params;
  this->spec = &spec;
  this->myrank = myrank;
  this->numprocs = numprocs;

  pi = 4.0 * atan(1.0);

  /* setting parameters */
  Kmin = para->dparams["kernel_K_Kmin"];
  Kmax = para->dparams["kernel_K_Kmax"];
  nK = para->iparams["kernel_K_nK"];

  qmin = para->dparams["fast_qmin"];
  qmax = para->dparams["fast_qmax"];
  mumin = para->dparams["fast_mumin"];
  mumax = para->dparams["fast_mumax"];
  phimin = para->dparams["fast_phimin"];
  phimax = para->dparams["fast_phimax"];

  nq = para->iparams["fast_nq"];
  nmu = para->iparams["fast_nmu"];
  nphi = para->iparams["fast_nphi"];

  kernel_root = para->sparams["kernel_root"];

  /* allocate memories */
  q = new double[nq];
  mu = new double[nmu];
  phi = new double[nphi];
  wq = new double[nq];
  wmu = new double[nmu];
  wphi = new double[nphi];

  L1 = new double[nK * nK * nq * 2 * 3];
  N2 = new double[nK * nK * nq * 2 * 3];
  T3 = new double[nK * nK * nq * 8];
  U3 = new double[nK * nK * nq * 8];
  V3 = new double[nK * nK * 8 * 3];

  t_q = gsl_integration_glfixed_table_alloc(nq);
  t_mu = gsl_integration_glfixed_table_alloc(nmu);
  t_phi = gsl_integration_glfixed_table_alloc(nphi);

  /* setting Gaussian quadrature */
  for (int i = 0; i < nq; i++) {
    gsl_integration_glfixed_point(log(qmin), log(qmax), i, &q[i], &wq[i], t_q);
    q[i] = exp(q[i]);
  }

  for (int i = 0; i < nmu; i++) {
    gsl_integration_glfixed_point(mumin, mumax, i, &mu[i], &wmu[i], t_mu);
  }

  for (int i = 0; i < nphi; i++) {
    gsl_integration_glfixed_point(phimin, phimax, i, &phi[i], &wphi[i], t_phi);
  }

  /* main calculation */
  set_K_bin();

  if (myrank == 0)
    save_linear_power();

  compute_diagram_bispectrum();

  // compute_kernels();
}

fast_kernels_bispec::~fast_kernels_bispec() {
  delete[] K;
  delete[] q;
  delete[] mu;
  delete[] phi;
  delete[] wq;
  delete[] wmu;
  delete[] wphi;
  delete[] L1;
  delete[] N2;
  delete[] T3;
  delete[] U3;
  delete[] V3;
  gsl_integration_glfixed_table_free(t_q);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_integration_glfixed_table_free(t_phi);
}

void fast_kernels_bispec::set_K_bin(void) {
  ifstream ifs;
  FILE *fp;
  char fname[256];

  K = new double[nK];

  for (int i = 0; i < nK; ++i) {
    K[i] = (log(Kmax) - log(Kmin)) / ((double)nK - 1.0) * i + log(Kmin);
    K[i] = exp(K[i]);
  }

  /* save K bin */
  sprintf(fname, "%s/Kbin.dat", kernel_root.c_str());

  if ((fp = fopen(fname, "wb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fwrite(&nK, sizeof(int), 1, fp);
  fwrite(K, sizeof(double), nK, fp);
  fwrite(&nK, sizeof(int), 1, fp);

  fclose(fp);

  return;
}

void fast_kernels_bispec::save_linear_power(void) {
  FILE *fp;
  char fname[256];
  int n;
  double kmin, kmax;
  double *k_, *P_;

  kmin = para->dparams["kmin"];
  kmax = para->dparams["kmax"];
  n = para->iparams["nint"];

  k_ = new double[n];
  P_ = new double[n];

  sprintf(fname, "%s/linear_power.dat", kernel_root.c_str());

  if ((fp = fopen(fname, "wb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fwrite(&n, sizeof(int), 1, fp);

  for (int i = 0; i < n; ++i) {
    k_[i] = (log(kmax) - log(kmin)) / ((double)n - 1.0) * i + log(kmin);
    k_[i] = exp(k_[i]);
    P_[i] = spec->P0(k_[i]);
  }

  fwrite(k_, sizeof(double), n, fp);
  fwrite(P_, sizeof(double), n, fp);

  fwrite(&n, sizeof(int), 1, fp);

  fclose(fp);
  delete[] k_;
  delete[] P_;

  return;
}

void fast_kernels_bispec::compute_diagram_bispectrum(void) {
  FILE *fp;
  char base[256], fname[256];
  int ind;
  double buf;
  double ki, mu12;
  double k1, k2, k3, qi, p1, p2, p3, r1, r2, r3;
  double Pk1, Pk2, Pk3, Pkq, Pkp1, Pkp2, Pkp3, Pkr1, Pkr2, Pkr3;
  double *P0, *G1_1l, *G2_1l, *F2, *Bcorr222, *Bcorr321;
  Type a, b, c;
  Vector kk1, kk2, kk3, qq, pp1, pp2, pp3, rr1, rr2, rr3;

  P0 = new double[3];
  G1_1l = new double[6];
  G2_1l = new double[6];
  F2 = new double[6];
  Bcorr222 = new double[8];
  Bcorr321 = new double[8];

  sprintf(base, "%s/diagram_bispec/diagram_bispec", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int iK1 = myrank; iK1 < nK; iK1 += numprocs) {
#else
  for (int iK1 = 0; iK1 < nK; ++iK1) {
#endif
    sprintf(fname, "%s_K%03d.dat", base, iK1);

    if ((fp = fopen(fname, "wb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fwrite(&iK1, sizeof(int), 1, fp);
    fwrite(&nK, sizeof(int), 1, fp);
    fwrite(&K[iK1], sizeof(double), 1, fp);

    for (int iK2 = 0; iK2 < nK; ++iK2) {
      for (int iK3 = 0; iK3 < nK; ++iK3) {
        k1 = 0.5 * (K[iK2] + K[iK3]);
        k2 = 0.5 * (K[iK3] + K[iK1]);
        k3 = 0.5 * (K[iK1] + K[iK2]);

        mu12 = -(sqr(k1) + sqr(k2) - sqr(k3)) / (2.0 * k1 * k2);

        kk1.x = 0.0, kk1.y = 0.0, kk1.z = k1;
        kk2.x = 0.0, kk2.y = k2 * sqrt(1.0 - sqr(mu12)), kk2.z = k2 * mu12;
        kk3 = -kk1 - kk2;

        Pk1 = fidP0(k1);
        Pk2 = fidP0(k2);
        Pk3 = fidP0(k3);

        for (int i = 0; i < 8; ++i) {
          Bcorr222[i] = 0.0;
          Bcorr321[i] = 0.0;
        }

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];
          Pkq = fidP0(qi);

          for (int imu = 0; imu < nmu; ++imu) {
            for (int iphi = 0; iphi < nphi; ++iphi) {
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

              Pkp1 = fidP0(p1);
              Pkp2 = fidP0(p2);
              Pkp3 = fidP0(p3);
              Pkr1 = fidP0(r1);
              Pkr2 = fidP0(r2);
              Pkr3 = fidP0(r3);

              for (int i1 = 0; i1 < 2; i1++) {
                for (int i2 = 0; i2 < 2; i2++) {
                  for (int i3 = 0; i3 < 2; i3++) {
                    a = (i1 == 0) ? DENS : VELO;
                    b = (i2 == 0) ? DENS : VELO;
                    c = (i3 == 0) ? DENS : VELO;
                    ind = 4 * i1 + 2 * i2 + 1 * i3;

                    // IR-safe integrand for Bk222
                    if (p1 > qi && r2 > qi)
                      Bcorr222[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          F2_sym(a, pp1, qq) * F2_sym(b, rr2, -qq) *
                          F2_sym(c, -rr2, -pp1) * Pkp1 * Pkq * Pkr2;

                    if (r1 > qi && p2 > qi)
                      Bcorr222[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          F2_sym(a, rr1, -qq) * F2_sym(b, pp2, qq) *
                          F2_sym(c, -pp2, -rr1) * Pkr1 * Pkq * Pkp2;

                    if (p3 > qi && r2 > qi)
                      Bcorr222[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          F2_sym(c, pp3, qq) * F2_sym(b, rr2, -qq) *
                          F2_sym(a, -rr2, -pp3) * Pkp3 * Pkq * Pkr2;

                    if (r3 > qi && p2 > qi)
                      Bcorr222[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          F2_sym(c, rr3, -qq) * F2_sym(b, pp2, qq) *
                          F2_sym(a, -pp2, -rr3) * Pkr3 * Pkq * Pkp2;

                    if (p1 > qi && r3 > qi)
                      Bcorr222[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          F2_sym(a, pp1, qq) * F2_sym(c, rr3, -qq) *
                          F2_sym(b, -rr3, -pp1) * Pkp1 * Pkq * Pkr3;

                    if (r1 > qi && p3 > qi)
                      Bcorr222[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          F2_sym(a, rr1, -qq) * F2_sym(c, pp3, qq) *
                          F2_sym(b, -pp3, -rr1) * Pkr1 * Pkq * Pkp3;

                    // IR-safe integrand for Bk321
                    if (p2 > qi)
                      Bcorr321[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          (F3_sym(a, -kk3, -pp2, -qq) * F2_sym(b, pp2, qq) *
                               Pkp2 * Pkq * Pk3 +
                           F3_sym(c, -kk1, -pp2, -qq) * F2_sym(b, pp2, qq) *
                               Pkp2 * Pkq * Pk1);

                    if (r2 > qi)
                      Bcorr321[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          (F3_sym(a, -kk3, -rr2, qq) * F2_sym(b, rr2, -qq) *
                               Pkr2 * Pkq * Pk3 +
                           F3_sym(c, -kk1, -rr2, qq) * F2_sym(b, rr2, -qq) *
                               Pkr2 * Pkq * Pk1);

                    if (p3 > qi)
                      Bcorr321[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          (F3_sym(a, -kk2, -pp3, -qq) * F2_sym(c, pp3, qq) *
                               Pkp3 * Pkq * Pk2 +
                           F3_sym(b, -kk1, -pp3, -qq) * F2_sym(c, pp3, qq) *
                               Pkp3 * Pkq * Pk1);

                    if (r3 > qi)
                      Bcorr321[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          (F3_sym(a, -kk2, -rr3, qq) * F2_sym(c, rr3, -qq) *
                               Pkr3 * Pkq * Pk2 +
                           F3_sym(b, -kk1, -rr3, qq) * F2_sym(c, rr3, -qq) *
                               Pkr3 * Pkq * Pk1);

                    if (p1 > qi)
                      Bcorr321[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          (F3_sym(b, -kk3, -pp1, -qq) * F2_sym(a, pp1, qq) *
                               Pkp1 * Pkq * Pk3 +
                           F3_sym(c, -kk2, -pp1, -qq) * F2_sym(a, pp1, qq) *
                               Pkp1 * Pkq * Pk2);

                    if (r1 > qi)
                      Bcorr321[ind] +=
                          cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                          (F3_sym(b, -kk3, -rr1, qq) * F2_sym(a, rr1, -qq) *
                               Pkr1 * Pkq * Pk3 +
                           F3_sym(c, -kk2, -rr1, qq) * F2_sym(a, rr1, -qq) *
                               Pkr1 * Pkq * Pk2);
                  }
                }
              }
            }
          }
        }

        // In Bcorr222, the additional division by 2 appears only in IR-safe
        // integrand. See Eq. B.13 of Baldauf et al. (2015)
        for (int i = 0; i < 8; ++i) {
          Bcorr222[i] *= 8.0 / cub(2.0 * pi) / 2.0;
          Bcorr321[i] *= 6.0 / cub(2.0 * pi);
        }

        P0[0] = fidP0(k1);
        P0[1] = fidP0(k2);
        P0[2] = fidP0(k3);

        G1_1l[0 + 3 * 0] = spec->Gamma1_1loop(DENS, k1);
        G1_1l[1 + 3 * 0] = spec->Gamma1_1loop(DENS, k2);
        G1_1l[2 + 3 * 0] = spec->Gamma1_1loop(DENS, k3);
        G1_1l[0 + 3 * 1] = spec->Gamma1_1loop(VELO, k1);
        G1_1l[1 + 3 * 1] = spec->Gamma1_1loop(VELO, k2);
        G1_1l[2 + 3 * 1] = spec->Gamma1_1loop(VELO, k3);

        G2_1l[0 + 3 * 0] = spec->Gamma2_1loop(DENS, k1, k2, k3);
        G2_1l[1 + 3 * 0] = spec->Gamma2_1loop(DENS, k2, k3, k1);
        G2_1l[2 + 3 * 0] = spec->Gamma2_1loop(DENS, k3, k1, k2);
        G2_1l[0 + 3 * 1] = spec->Gamma2_1loop(VELO, k1, k2, k3);
        G2_1l[1 + 3 * 1] = spec->Gamma2_1loop(VELO, k2, k3, k1);
        G2_1l[2 + 3 * 1] = spec->Gamma2_1loop(VELO, k3, k1, k2);

        F2[0 + 3 * 0] = F2_sym(DENS, kk1, kk2);
        F2[1 + 3 * 0] = F2_sym(DENS, kk2, kk3);
        F2[2 + 3 * 0] = F2_sym(DENS, kk3, kk1);
        F2[0 + 3 * 1] = F2_sym(VELO, kk1, kk2);
        F2[1 + 3 * 1] = F2_sym(VELO, kk2, kk3);
        F2[2 + 3 * 1] = F2_sym(VELO, kk3, kk1);

        fwrite(P0, sizeof(double), 3, fp);
        fwrite(G1_1l, sizeof(double), 6, fp);
        fwrite(G2_1l, sizeof(double), 6, fp);
        fwrite(F2, sizeof(double), 6, fp);
        fwrite(Bcorr222, sizeof(double), 8, fp);
        fwrite(Bcorr321, sizeof(double), 8, fp);
      }
    }

    fwrite(&iK1, sizeof(int), 1, fp);
    fwrite(&nK, sizeof(int), 1, fp);

    fclose(fp);
  }

  delete[] P0;
  delete[] G1_1l;
  delete[] G2_1l;
  delete[] F2;
  delete[] Bcorr222;
  delete[] Bcorr321;

  printf("#diagram bispectrum done\n");

  return;
}

void fast_kernels_bispec::compute_kernels(void) {
  L1_kernel();
  printf("#L1 done:%d\n", myrank);

  N2_kernel();
  printf("#N2 done:%d\n", myrank);

  T3U3V3_kernel();
  printf("#T3, U3, V3 done:%d\n", myrank);

  return;
}

void fast_kernels_bispec::save_kernel_data(char *base, double *data, int size,
                                           int ik) {
  FILE *fp;
  char fname[256];

  sprintf(fname, "%s_K%03d.dat", base, ik);

  if ((fp = fopen(fname, "wb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fwrite(data, sizeof(double), size, fp);

  fclose(fp);

  return;
}

double fast_kernels_bispec::fidP0(double k) { return spec->P0(k); }

void fast_kernels_bispec::L1_kernel(void) {
  char base[256];
  double k1, k2, k3, qi;

  sprintf(base, "%s/L1/L1", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int iK1 = myrank; iK1 < nK; iK1 += numprocs) {
#else
  for (int iK1 = 0; iK1 < nK; ++iK1) {
#endif
    for (int iK2 = 0; iK2 < nK; ++iK2) {
      for (int iK3 = 0; iK3 < nK; ++iK3) {
        k1 = 0.5 * (K[iK2] + K[iK3]);
        k2 = 0.5 * (K[iK3] + K[iK1]);
        k3 = 0.5 * (K[iK1] + K[iK2]);

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];
          L1[0 + 3 * (0 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma1_1loop(DENS, k1, qi);
          L1[1 + 3 * (0 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma1_1loop(DENS, k2, qi);
          L1[2 + 3 * (0 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma1_1loop(DENS, k3, qi);
          L1[0 + 3 * (1 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma1_1loop(VELO, k1, qi);
          L1[1 + 3 * (1 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma1_1loop(VELO, k2, qi);
          L1[2 + 3 * (1 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma1_1loop(VELO, k3, qi);
        }
      }
    }
    save_kernel_data(base, L1, nK * nK * nq * 2 * 3, iK1);
  }

  return;
}

void fast_kernels_bispec::N2_kernel(void) {
  char base_N2[256];
  double qi, k1, k2, k3;

  sprintf(base_N2, "%s/N2/N2", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int iK1 = myrank; iK1 < nK; iK1 += numprocs) {
#else
  for (int iK1 = 0; iK1 < nK; ++iK1) {
#endif
    for (int iK2 = 0; iK2 < nK; ++iK2) {
      for (int iK3 = 0; iK3 < nK; ++iK3) {
        k1 = 0.5 * (K[iK2] + K[iK3]);
        k2 = 0.5 * (K[iK3] + K[iK1]);
        k3 = 0.5 * (K[iK1] + K[iK2]);

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];
          N2[0 + 3 * (0 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma2_1loop(DENS, k1, k2, k3, qi) / (4.0 * pi);
          N2[1 + 3 * (0 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma2_1loop(DENS, k2, k3, k1, qi) / (4.0 * pi);
          N2[2 + 3 * (0 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma2_1loop(DENS, k3, k1, k2, qi) / (4.0 * pi);
          N2[0 + 3 * (1 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma2_1loop(VELO, k1, k2, k3, qi) / (4.0 * pi);
          N2[1 + 3 * (1 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma2_1loop(VELO, k2, k3, k1, qi) / (4.0 * pi);
          N2[2 + 3 * (1 + 2 * (iq + nq * (iK3 + nK * iK2)))] =
              kernel_Gamma2_1loop(VELO, k3, k1, k2, qi) / (4.0 * pi);
        }
      }
    }
    save_kernel_data(base_N2, N2, nK * nK * nq * 2 * 3, iK1);
  }

  return;
}

void fast_kernels_bispec::T3U3V3_kernel(void) {
  char base_T3[256], base_U3[256], base_V3[256];
  int ind;
  double qi, k1, k2, k3, mu12, p1, p2, p3, r1, r2, r3;
  double Pkq, Pk1, Pk2, Pk3, Pkp1, Pkp2, Pkp3, Pkr1, Pkr2, Pkr3;
  double res_T3[8], res_U3[8], res_V3[3 * 8];
  Type a, b, c;
  Vector kk1, kk2, kk3, pp1, pp2, pp3, rr1, rr2, rr3, qq;

  sprintf(base_T3, "%s/T3/T3", kernel_root.c_str());
  sprintf(base_U3, "%s/U3/U3", kernel_root.c_str());
  sprintf(base_V3, "%s/V3/V3", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int iK1 = myrank; iK1 < nK; iK1 += numprocs) {
#else
  for (int iK1 = 0; iK1 < nK; ++iK1) {
#endif
    for (int iK2 = 0; iK2 < nK; ++iK2) {
      for (int iK3 = 0; iK3 < nK; ++iK3) {
        k1 = 0.5 * (K[iK2] + K[iK3]);
        k2 = 0.5 * (K[iK3] + K[iK1]);
        k3 = 0.5 * (K[iK1] + K[iK2]);

        mu12 = -(sqr(k1) + sqr(k2) - sqr(k3)) / (2.0 * k1 * k2);
        kk1.x = 0.0, kk1.y = 0.0, kk1.z = k1;
        kk2.x = 0.0, kk2.y = k2 * sqrt(1.0 - sqr(mu12)), kk2.z = k2 * mu12;
        kk3 = -kk1 - kk2;

        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 3; ++j) {
            res_V3[j + 3 * i] = 0.0;
          }
        }

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];
          Pkq = fidP0(qi);

          for (int i = 0; i < 8; ++i) {
            res_T3[i] = 0.0;
            res_U3[i] = 0.0;
          }

          for (int imu = 0; imu < nmu; ++imu) {
            for (int iphi = 0; iphi < nphi; ++iphi) {
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

              Pkp1 = fidP0(p1);
              Pkp2 = fidP0(p2);
              Pkp3 = fidP0(p3);
              Pkr1 = fidP0(r1);
              Pkr2 = fidP0(r2);
              Pkr3 = fidP0(r3);

              Pk1 = fidP0(k1);
              Pk2 = fidP0(k2);
              Pk3 = fidP0(k3);

              for (int i1 = 0; i1 < 2; i1++) {
                for (int i2 = 0; i2 < 2; i2++) {
                  for (int i3 = 0; i3 < 2; i3++) {
                    a = (i1 == 0) ? DENS : VELO;
                    b = (i2 == 0) ? DENS : VELO;
                    c = (i3 == 0) ? DENS : VELO;
                    ind = 4 * i1 + 2 * i2 + 1 * i3;

                    res_T3[ind] += wmu[imu] * wphi[iphi] *
                                   (F2_sym(a, pp1, qq) * F2_sym(b, rr2, -qq) *
                                        F2_sym(c, -rr2, -pp1) * Pkp1 * Pkr2 +
                                    F2_sym(a, pp1, qq) * F2_sym(b, -rr3, -pp1) *
                                        F2_sym(c, rr3, -qq) * Pkp1 * Pkr3 +
                                    F2_sym(a, -rr3, -pp2) * F2_sym(b, qq, pp2) *
                                        F2_sym(c, rr3, -qq) * Pkp2 * Pkr3);
                    res_U3[ind] += wmu[imu] * wphi[iphi] *
                                   (F3_sym(a, -kk3, -pp2, -qq) *
                                        F2_sym(b, pp2, qq) * Pkp2 * Pk3 +
                                    F3_sym(a, -kk2, -pp3, -qq) *
                                        F2_sym(c, pp3, qq) * Pkp3 * Pk2 +
                                    F3_sym(b, -qq, -pp1, -kk3) *
                                        F2_sym(a, pp1, qq) * Pkp1 * Pk3 +
                                    F3_sym(b, -kk1, -pp3, -qq) *
                                        F2_sym(c, pp3, qq) * Pkp3 * Pk1 +
                                    F3_sym(c, -pp1, -qq, -kk2) *
                                        F2_sym(a, pp1, qq) * Pkp1 * Pk2 +
                                    F3_sym(c, -kk1, -qq, -pp2) *
                                        F2_sym(b, pp2, qq) * Pkp2 * Pk1);
                    res_V3[0 + 3 * ind] +=
                        cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                        (F3_sym(b, -kk1, -pp3, -qq) * F2_sym(c, pp3, qq) *
                             Pkp3 * Pkq +
                         F3_sym(c, -kk1, -qq, -pp2) * F2_sym(b, pp2, qq) *
                             Pkp2 * Pkq);
                    res_V3[1 + 3 * ind] +=
                        cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                        (F3_sym(a, -kk2, -pp3, -qq) * F2_sym(c, pp3, qq) *
                             Pkp3 * Pkq +
                         F3_sym(c, -pp1, -qq, -kk2) * F2_sym(a, pp1, qq) *
                             Pkp1 * Pkq);
                    res_V3[2 + 3 * ind] +=
                        cub(qi) * wq[iq] * wmu[imu] * wphi[iphi] *
                        (F3_sym(a, -kk3, -pp2, -qq) * F2_sym(b, pp2, qq) *
                             Pkp2 * Pkq +
                         F3_sym(b, -qq, -pp1, -kk3) * F2_sym(a, pp1, qq) *
                             Pkp1 * Pkq);
                  }
                }
              }
            }
          }
          for (int i = 0; i < 8; ++i) {
            T3[i + 8 * (iq + nq * (iK3 + nK * iK2))] = res_T3[i] / (4.0 * pi);
            U3[i + 8 * (iq + nq * (iK3 + nK * iK2))] = res_U3[i] / (4.0 * pi);
          }
        }
        for (int i = 0; i < 8; ++i) {
          V3[0 + 3 * (i + 8 * (iK3 + nK * iK2))] =
              res_V3[0 + 3 * i] / cub(2.0 * pi);
          V3[1 + 3 * (i + 8 * (iK3 + nK * iK2))] =
              res_V3[1 + 3 * i] / cub(2.0 * pi);
          V3[2 + 3 * (i + 8 * (iK3 + nK * iK2))] =
              res_V3[2 + 3 * i] / cub(2.0 * pi);
        }
      }
    }

    save_kernel_data(base_T3, T3, nK * nK * nq * 8, iK1);
    save_kernel_data(base_U3, U3, nK * nK * nq * 8, iK1);
    save_kernel_data(base_V3, V3, nK * nK * 8 * 3, iK1);
  }

  return;
}
