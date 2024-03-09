#include "fast_kernels.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_spline.h>
#include <iostream>

fast_kernels::fast_kernels(params &params, spectra &spec, int myrank,
                           int numprocs) {
  this->para = &params;
  this->spec = &spec;
  this->myrank = myrank;
  this->numprocs = numprocs;

  pi = 4.0 * atan(1.0);

  /* setting parameters */
  qmin = para->dparams["fast_qmin"];
  qmax = para->dparams["fast_qmax"];
  pmin = para->dparams["fast_qmin"];
  pmax = para->dparams["fast_qmax"];
  mumin = para->dparams["fast_mumin"];
  mumax = para->dparams["fast_mumax"];
  phimin = para->dparams["fast_phimin"];
  phimax = para->dparams["fast_phimax"];

  nq = para->iparams["fast_nq"];
  np = para->iparams["fast_nq"];
  nmu = para->iparams["fast_nmu"];
  nphi = para->iparams["fast_nphi"];
  nr = para->iparams["fast_nr"];
  nx = para->iparams["fast_nx"];

  kernel_root = para->sparams["kernel_root"];

  /* allocate memories */
  q = new double[nq];
  p = new double[np];
  mu = new double[nmu];
  phi = new double[nphi];
  wq = new double[nq];
  wp = new double[np];
  wmu = new double[nmu];
  wphi = new double[nphi];

  L1 = new double[nq * 2 + nr * nx * nq * 2 * 3];
  M1 = new double[nq * 2];
  X2 = new double[nq * 4];
  Y2 = new double[nq * 4];
  Z2 = new double[nq * 4];
  Q2 = new double[nq * 4];
  R2 = new double[nq * 4];
  S3 = new double[nq * 4];
  N2 = new double[nr * nx * nq * 2 * 3];
  T3 = new double[nr * nx * nq * 8];
  U3 = new double[nr * nx * nq * 8];
  V3 = new double[nr * nx * 8 * 3];

  t_q = gsl_integration_glfixed_table_alloc(nq);
  t_p = gsl_integration_glfixed_table_alloc(np);
  t_mu = gsl_integration_glfixed_table_alloc(nmu);
  t_phi = gsl_integration_glfixed_table_alloc(nphi);
  t_r = gsl_integration_glfixed_table_alloc(nr);
  t_x = gsl_integration_glfixed_table_alloc(nx);

  /* setting Gaussian quadrature */
  for (int i = 0; i < nq; i++) {
    gsl_integration_glfixed_point(log(qmin), log(qmax), i, &q[i], &wq[i], t_q);
    q[i] = exp(q[i]);
  }

  for (int i = 0; i < np; i++) {
    gsl_integration_glfixed_point(log(pmin), log(pmax), i, &p[i], &wp[i], t_p);
    p[i] = exp(p[i]);
  }

  for (int i = 0; i < nmu; i++) {
    gsl_integration_glfixed_point(mumin, mumax, i, &mu[i], &wmu[i], t_mu);
  }

  for (int i = 0; i < nphi; i++) {
    gsl_integration_glfixed_point(phimin, phimax, i, &phi[i], &wphi[i], t_phi);
  }

  /* main calculation */
  set_k_bin();
  if (myrank == 0)
    save_linear_power();
  compute_diagram_spectrum();
  compute_diagram_bispectrum();
  compute_kernels();
}

fast_kernels::~fast_kernels() {
  delete[] k;
  delete[] q;
  delete[] p;
  delete[] mu;
  delete[] phi;
  delete[] wq;
  delete[] wp;
  delete[] wmu;
  delete[] wphi;
  delete[] L1;
  delete[] M1;
  delete[] X2;
  delete[] Y2;
  delete[] Z2;
  delete[] Q2;
  delete[] R2;
  delete[] S3;
  delete[] N2;
  delete[] T3;
  delete[] U3;
  delete[] V3;
  gsl_integration_glfixed_table_free(t_q);
  gsl_integration_glfixed_table_free(t_p);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_integration_glfixed_table_free(t_phi);
  gsl_integration_glfixed_table_free(t_r);
  gsl_integration_glfixed_table_free(t_x);
}

void fast_kernels::set_k_bin(void) {
  ifstream ifs;
  FILE *fp;
  char fname[256];
  int lines, count;
  double kmin, kmax;
  string str, spacing, k_file_name;
  bool from_file;

  kmin = para->dparams["kernel_k_kmin"];
  kmax = para->dparams["kernel_k_kmax"];
  nk = para->iparams["kernel_k_nk"];
  from_file = para->bparams["kernel_k_from_file"];
  spacing = para->sparams["kernel_k_spacing"];
  k_file_name = para->sparams["kernel_k_file_name"];

  if (from_file) {
    ifs.open(k_file_name, ios::in);
    if (ifs.fail()) {
      cerr << "[ERROR] kernel k file open error:" << k_file_name << endl;
      exit(1);
    }

    lines = 0;
    while (getline(ifs, str)) {
      if (str[0] != '#')
        lines++;
    }
    nk = lines;

    k = new double[nk];

    ifs.clear();
    ifs.seekg(0, ios_base::beg);

    count = 0;
    while (getline(ifs, str)) {
      if (str[0] != '#') {
        sscanf(str.data(), "%lf", &k[count]);
        count++;
      }
    }

    ifs.close();
  } else {
    k = new double[nk];

    if (spacing == "linear") {
      for (int i = 0; i < nk; ++i) {
        k[i] = (kmax - kmin) / ((double)nk - 1.0) * i + kmin;
      }
    } else if (spacing == "log") {
      for (int i = 0; i < nk; ++i) {
        k[i] = (log(kmax) - log(kmin)) / ((double)nk - 1.0) * i + log(kmin);
        k[i] = exp(k[i]);
      }
    }
  }

  /* save k bin */
  sprintf(fname, "%s/kbin.dat", kernel_root.c_str());

  if ((fp = fopen(fname, "wb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fwrite(&nk, sizeof(int), 1, fp);

  fwrite(k, sizeof(double), nk, fp);

  fwrite(&nk, sizeof(int), 1, fp);

  fclose(fp);

  return;
}

void fast_kernels::save_linear_power(void) {
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

void fast_kernels::compute_diagram_spectrum(void) {
  FILE *fp;
  char base[256], fname[256];
  double buf;

  sprintf(base, "%s/diagram_spec/diagram_spec", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    sprintf(fname, "%s_k%03d.dat", base, ik);

    if ((fp = fopen(fname, "wb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fwrite(&ik, sizeof(int), 1, fp);
    fwrite(&nk, sizeof(int), 1, fp);

    buf = k[ik];
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->P0(k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->Gamma1_1loop(DENS, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Gamma1_1loop(VELO, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->Gamma1_2loop(DENS, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Gamma1_2loop(VELO, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->Pcorr2(DENS, DENS, TREE_TREE, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr2(DENS, VELO, TREE_TREE, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr2(VELO, VELO, TREE_TREE, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->Pcorr2(DENS, DENS, TREE_ONELOOP, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr2(DENS, VELO, TREE_ONELOOP, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr2(VELO, VELO, TREE_ONELOOP, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->Pcorr2(DENS, DENS, ONELOOP_ONELOOP, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr2(DENS, VELO, ONELOOP_ONELOOP, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr2(VELO, VELO, ONELOOP_ONELOOP, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    buf = spec->Pcorr3(DENS, DENS, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr3(DENS, VELO, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);
    buf = spec->Pcorr3(VELO, VELO, k[ik]);
    fwrite(&buf, sizeof(double), 1, fp);

    fwrite(&ik, sizeof(int), 1, fp);
    fwrite(&nk, sizeof(int), 1, fp);

    fclose(fp);
  }

  printf("#diagram of power spectrum done\n");

  return;
}

void fast_kernels::compute_diagram_bispectrum(void) {
  FILE *fp;
  char base[256], fname[256];
  int ind;
  double buf;
  double ki, ri, xi, p_i, kpi, mu12, rmin, rmax, xmin, xmax, wr, wx;
  double G1_1l_k1, G1_1l_k2, G1_1l_k3, G2_1l_k1_k2, G2_1l_k2_k3, G2_1l_k3_k1;
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
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    sprintf(fname, "%s_k%03d.dat", base, ik);

    if ((fp = fopen(fname, "wb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    ki = k[ik];

    fwrite(&ik, sizeof(int), 1, fp);
    fwrite(&nk, sizeof(int), 1, fp);
    fwrite(&nr, sizeof(int), 1, fp);
    fwrite(&nx, sizeof(int), 1, fp);
    fwrite(&ki, sizeof(double), 1, fp);

    for (int ir = 0; ir < nr; ++ir) {
      rmin = qmin / ki;
      rmax = qmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);
        kpi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p_i = ki * ri;

        k1 = p_i;
        k2 = kpi;
        k3 = ki;

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

        fwrite(&ri, sizeof(double), 1, fp);
        fwrite(&xi, sizeof(double), 1, fp);
        fwrite(P0, sizeof(double), 3, fp);
        fwrite(G1_1l, sizeof(double), 6, fp);
        fwrite(G2_1l, sizeof(double), 6, fp);
        fwrite(F2, sizeof(double), 6, fp);
        fwrite(Bcorr222, sizeof(double), 8, fp);
        fwrite(Bcorr321, sizeof(double), 8, fp);
      }
    }

    fwrite(&ik, sizeof(int), 1, fp);
    fwrite(&nk, sizeof(int), 1, fp);
    fwrite(&nr, sizeof(int), 1, fp);
    fwrite(&nx, sizeof(int), 1, fp);

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

void fast_kernels::compute_kernels(void) {
  L1_kernel();
  printf("#L1 done:%d\n", myrank);

  M1_kernel();
  printf("#M1 done:%d\n", myrank);

  X2Y2Z2_kernel();
  printf("#X2, Y2, Z2 done:%d\n", myrank);

  Q2R2_kernel();
  printf("#Q2, R2 done:%d\n", myrank);

  S3_kernel();
  printf("#S3 done:%d\n", myrank);

  N2_kernel();
  printf("#N2 done:%d\n", myrank);

  T3U3V3_kernel();
  printf("#T3, U3, V3 done:%d\n", myrank);

  return;
}

void fast_kernels::save_kernel_data(char *base, double *data, int size,
                                    int ik) {
  FILE *fp;
  char fname[256];

  sprintf(fname, "%s_k%03d.dat", base, ik);

  if ((fp = fopen(fname, "wb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fwrite(data, sizeof(double), size, fp);

  fclose(fp);

  return;
}

double fast_kernels::fidP0(double k) { return spec->P0(k); }

void fast_kernels::L1_kernel(void) {
  char base[256];
  int offset;
  double ki, qi, ri, xi, wr, wx;
  double rmin, rmax, xmin, xmax;
  double kpi, p_i, k1, k2, k3;

  sprintf(base, "%s/L1/L1", kernel_root.c_str());

  offset = 2 * nq;

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];

    for (int iq = 0; iq < nq; ++iq) {
      qi = q[iq];
      L1[0 + 2 * iq] = kernel_Gamma1_1loop(DENS, ki, qi);
      L1[1 + 2 * iq] = kernel_Gamma1_1loop(VELO, ki, qi);
    }

    for (int ir = 0; ir < nr; ++ir) {
      rmin = qmin / ki;
      rmax = qmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);
        kpi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p_i = ki * ri;

        k1 = p_i;
        k2 = kpi;
        k3 = ki;

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];
          L1[offset + 0 + 3 * (0 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma1_1loop(DENS, k1, qi);
          L1[offset + 1 + 3 * (0 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma1_1loop(DENS, k2, qi);
          L1[offset + 2 + 3 * (0 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma1_1loop(DENS, k3, qi);
          L1[offset + 0 + 3 * (1 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma1_1loop(VELO, k1, qi);
          L1[offset + 1 + 3 * (1 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma1_1loop(VELO, k2, qi);
          L1[offset + 2 + 3 * (1 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma1_1loop(VELO, k3, qi);
        }
      }
    }

    save_kernel_data(base, L1, nq * 2 + nr * nx * nq * 2 * 3, ik);
  }

  return;
}

void fast_kernels::M1_kernel(void) {
  char base[256];
  double ki, qi, p_i, wpi, Ppi, M1_d, M1_v;
  double G1_1l_d, G1_1l_v;

  sprintf(base, "%s/M1/M1", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];
    G1_1l_d = spec->Gamma1_1loop(DENS, ki);
    G1_1l_v = spec->Gamma1_1loop(VELO, ki);
    for (int iq = 0; iq < nq; ++iq) {
      qi = q[iq];
      M1_d = 0.0, M1_v = 0.0;
      for (int ip = 0; ip < np; ip++) {
        p_i = p[ip];
        wpi = wp[ip];
        Ppi = fidP0(p_i);
        M1_d += wpi * cub(p_i) * kernel_Gamma1_2loop(DENS, ki, p_i, qi) * Ppi;
        M1_v += wpi * cub(p_i) * kernel_Gamma1_2loop(VELO, ki, p_i, qi) * Ppi;
      }
      M1_d = M1_d / (2.0 * sqr(pi));
      M1_v = M1_v / (2.0 * sqr(pi));

      M1_d += 0.5 * G1_1l_d * kernel_Gamma1_1loop(DENS, ki, qi);
      M1_v += 0.5 * G1_1l_d * kernel_Gamma1_1loop(VELO, ki, qi);

      M1[0 + 2 * iq] = M1_d;
      M1[1 + 2 * iq] = M1_v;
    }
    save_kernel_data(base, M1, nq * 2, ik);
  }

  return;
}

void fast_kernels::X2Y2Z2_kernel(void) {
  char base_X2[256], base_Y2[256], base_Z2[256];
  double ki, qi, mui, wmui, kqi, Pkqi;
  double F2_d, F2_v, G2_1l_d, G2_1l_v;
  double X2_dd, X2_dv, X2_vd, X2_vv;
  double Y2_dd, Y2_dv, Y2_vd, Y2_vv;
  double Z2_dd, Z2_dv, Z2_vd, Z2_vv;
  Vector kk, qq;

  sprintf(base_X2, "%s/X2/X2", kernel_root.c_str());
  sprintf(base_Y2, "%s/Y2/Y2", kernel_root.c_str());
  sprintf(base_Z2, "%s/Z2/Z2", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];
    kk.x = ki, kk.y = 0.0, kk.z = 0.0;
    for (int iq = 0; iq < nq; ++iq) {
      qi = q[iq];
      X2_dd = 0.0, X2_dv = 0.0, X2_vd = 0.0, X2_vv = 0.0;
      Y2_dd = 0.0, Y2_dv = 0.0, Y2_vd = 0.0, Y2_vv = 0.0;
      Z2_dd = 0.0, Z2_dv = 0.0, Z2_vd = 0.0, Z2_vv = 0.0;
      for (int imu = 0; imu < nmu; imu++) {
        mui = mu[imu];
        wmui = wmu[imu];
        qq.x = qi * mui, qq.y = qi * sqrt(1.0 - sqr(mui)), qq.z = 0.0;
        kqi = sqrt((kk - qq) * (kk - qq));

        Pkqi = fidP0(kqi);
        F2_d = F2_sym(DENS, qq, kk - qq);
        F2_v = F2_sym(VELO, qq, kk - qq);
        G2_1l_d = spec->Gamma2_1loop(DENS, qi, kqi, ki);
        G2_1l_v = spec->Gamma2_1loop(VELO, qi, kqi, ki);

        X2_dd += wmui * F2_d * F2_d * Pkqi;
        X2_dv += wmui * F2_d * F2_v * Pkqi;
        X2_vd += wmui * F2_v * F2_d * Pkqi;
        X2_vv += wmui * F2_v * F2_v * Pkqi;

        Y2_dd += wmui * F2_d * G2_1l_d * Pkqi;
        Y2_dv += wmui * F2_d * G2_1l_v * Pkqi;
        Y2_vd += wmui * F2_v * G2_1l_d * Pkqi;
        Y2_vv += wmui * F2_v * G2_1l_v * Pkqi;

        Z2_dd += wmui * G2_1l_d * G2_1l_d * Pkqi;
        Z2_dv += wmui * G2_1l_d * G2_1l_v * Pkqi;
        Z2_vd += wmui * G2_1l_v * G2_1l_d * Pkqi;
        Z2_vv += wmui * G2_1l_v * G2_1l_v * Pkqi;
      }
      X2[0 + 4 * iq] = X2_dd / 2.0;
      X2[1 + 4 * iq] = X2_dv / 2.0;
      X2[2 + 4 * iq] = X2_vd / 2.0;
      X2[3 + 4 * iq] = X2_vv / 2.0;

      Y2[0 + 4 * iq] = Y2_dd / 2.0;
      Y2[1 + 4 * iq] = Y2_dv / 2.0;
      Y2[2 + 4 * iq] = Y2_vd / 2.0;
      Y2[3 + 4 * iq] = Y2_vv / 2.0;

      Z2[0 + 4 * iq] = Z2_dd / 2.0;
      Z2[1 + 4 * iq] = Z2_dv / 2.0;
      Z2[2 + 4 * iq] = Z2_vd / 2.0;
      Z2[3 + 4 * iq] = Z2_vv / 2.0;
    }
    save_kernel_data(base_X2, X2, nq * 4, ik);
    save_kernel_data(base_Y2, Y2, nq * 4, ik);
    save_kernel_data(base_Z2, Z2, nq * 4, ik);
  }

  return;
}

void fast_kernels::Q2R2_kernel(void) {
  char base_Q2[256], base_R2[256];
  double ki, qi, p_i, kpi, Ppi, Pkpi;
  double kernel_G2_1l_d, kernel_G2_1l_v, F2_d, F2_v, G2_1l_d, G2_1l_v;
  double Q2_dd, Q2_dv, Q2_vd, Q2_vv;
  double R2_dd, R2_dv, R2_vd, R2_vv;
  Vector kk, pp;

  sprintf(base_Q2, "%s/Q2/Q2", kernel_root.c_str());
  sprintf(base_R2, "%s/R2/R2", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];
    kk.x = 0.0, kk.y = 0.0, kk.z = ki;
    for (int iq = 0; iq < nq; ++iq) {
      qi = q[iq];
      Q2_dd = 0.0, Q2_dv = 0.0, Q2_vd = 0.0, Q2_vv = 0.0;
      R2_dd = 0.0, R2_dv = 0.0, R2_vd = 0.0, R2_vv = 0.0;
      for (int ip = 0; ip < np; ++ip) {
        for (int imu = 0; imu < nmu; ++imu) {
          for (int iphi = 0; iphi < nphi; ++iphi) {
            p_i = p[ip];
            pp.x = p_i * sqrt(1.0 - sqr(mu[imu])) * cos(phi[iphi]);
            pp.y = p_i * sqrt(1.0 - sqr(mu[imu])) * sin(phi[iphi]);
            pp.z = p_i * mu[imu];
            kpi = sqrt((kk - pp) * (kk - pp));

            Ppi = fidP0(p_i);
            Pkpi = fidP0(kpi);

            kernel_G2_1l_d =
                kernel_Gamma2_1loop(DENS, p_i, kpi, ki, qi) / (4.0 * pi);
            kernel_G2_1l_v =
                kernel_Gamma2_1loop(VELO, p_i, kpi, ki, qi) / (4.0 * pi);

            F2_d = F2_sym(DENS, pp, kk - pp);
            F2_v = F2_sym(VELO, pp, kk - pp);

            G2_1l_d = spec->Gamma2_1loop(DENS, p_i, kpi, ki);
            G2_1l_v = spec->Gamma2_1loop(VELO, p_i, kpi, ki);

            Q2_dd += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * F2_d *
                     kernel_G2_1l_d * Pkpi * Ppi;
            Q2_dv += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * F2_d *
                     kernel_G2_1l_v * Pkpi * Ppi;
            Q2_vd += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * F2_v *
                     kernel_G2_1l_d * Pkpi * Ppi;
            Q2_vv += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * F2_v *
                     kernel_G2_1l_v * Pkpi * Ppi;

            R2_dd += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * G2_1l_d *
                     kernel_G2_1l_d * Pkpi * Ppi;
            R2_dv += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * G2_1l_d *
                     kernel_G2_1l_v * Pkpi * Ppi;
            R2_vd += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * G2_1l_v *
                     kernel_G2_1l_d * Pkpi * Ppi;
            R2_vv += cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] * G2_1l_v *
                     kernel_G2_1l_v * Pkpi * Ppi;
          }
        }
      }
      Q2[0 + 4 * iq] = Q2_dd / cub(2.0 * pi);
      Q2[1 + 4 * iq] = Q2_dv / cub(2.0 * pi);
      Q2[2 + 4 * iq] = Q2_vd / cub(2.0 * pi);
      Q2[3 + 4 * iq] = Q2_vv / cub(2.0 * pi);

      R2[0 + 4 * iq] = R2_dd / cub(2.0 * pi);
      R2[1 + 4 * iq] = R2_dv / cub(2.0 * pi);
      R2[2 + 4 * iq] = R2_vd / cub(2.0 * pi);
      R2[3 + 4 * iq] = R2_vv / cub(2.0 * pi);
    }
    save_kernel_data(base_Q2, Q2, nq * 4, ik);
    save_kernel_data(base_R2, R2, nq * 4, ik);
  }

  return;
}

void fast_kernels::S3_kernel(void) {
  char base_S3[256];
  double ki, qi, p_i, kpqi, Ppi, Pkpqi;
  double F3_d, F3_v;
  double S3_dd, S3_dv, S3_vd, S3_vv;
  Vector kk, pp, qq;

  sprintf(base_S3, "%s/S3/S3", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];
    kk.x = 0.0, kk.y = 0.0, kk.z = ki;
    for (int iq = 0; iq < nq; ++iq) {
      qi = q[iq];
      S3_dd = 0.0, S3_dv = 0.0, S3_vd = 0.0, S3_vv = 0.0;
      for (int imuq = 0; imuq < nmu; ++imuq) {
        qq.x = 0.0, qq.y = qi * sqrt(1.0 - sqr(mu[imuq])), qq.z = qi * mu[imuq];
        for (int ip = 0; ip < np; ++ip) {
          for (int imu = 0; imu < nmu; ++imu) {
            for (int iphi = 0; iphi < nphi; ++iphi) {
              p_i = p[ip];
              pp.x = p_i * sqrt(1.0 - sqr(mu[imu])) * cos(phi[iphi]);
              pp.y = p_i * sqrt(1.0 - sqr(mu[imu])) * sin(phi[iphi]);
              pp.z = p_i * mu[imu];
              kpqi = sqrt((kk - pp - qq) * (kk - pp - qq));

              Ppi = fidP0(p_i);
              Pkpqi = fidP0(kpqi);

              F3_d = F3_sym(DENS, pp, qq, kk - pp - qq);
              F3_v = F3_sym(VELO, pp, qq, kk - pp - qq);

              S3_dd += wmu[imuq] * cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] *
                       F3_d * F3_d * Pkpqi * Ppi;
              S3_dv += wmu[imuq] * cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] *
                       F3_d * F3_v * Pkpqi * Ppi;
              S3_vd += wmu[imuq] * cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] *
                       F3_v * F3_d * Pkpqi * Ppi;
              S3_vv += wmu[imuq] * cub(p_i) * wp[ip] * wphi[iphi] * wmu[imu] *
                       F3_v * F3_v * Pkpqi * Ppi;
            }
          }
        }
      }
      S3[0 + 4 * iq] = S3_dd / cub(2.0 * pi) / 2.0;
      S3[1 + 4 * iq] = S3_dv / cub(2.0 * pi) / 2.0;
      S3[2 + 4 * iq] = S3_vd / cub(2.0 * pi) / 2.0;
      S3[3 + 4 * iq] = S3_vv / cub(2.0 * pi) / 2.0;
    }
    save_kernel_data(base_S3, S3, nq * 4, ik);
  }

  return;
}

void fast_kernels::N2_kernel(void) {
  char base_N2[256];
  double ki, ri, xi, p_i, kpi, rmin, rmax, xmin, xmax, wr, wx;
  double qi, k1, k2, k3;

  sprintf(base_N2, "%s/N2/N2", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];
    for (int ir = 0; ir < nr; ++ir) {
      rmin = qmin / ki;
      rmax = qmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);
        kpi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p_i = ki * ri;

        k1 = p_i;
        k2 = kpi;
        k3 = ki;

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];
          N2[0 + 3 * (0 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma2_1loop(DENS, k1, k2, k3, qi) / (4.0 * pi);
          N2[1 + 3 * (0 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma2_1loop(DENS, k2, k3, k1, qi) / (4.0 * pi);
          N2[2 + 3 * (0 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma2_1loop(DENS, k3, k1, k2, qi) / (4.0 * pi);
          N2[0 + 3 * (1 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma2_1loop(VELO, k1, k2, k3, qi) / (4.0 * pi);
          N2[1 + 3 * (1 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma2_1loop(VELO, k2, k3, k1, qi) / (4.0 * pi);
          N2[2 + 3 * (1 + 2 * (iq + nq * (ix + nx * ir)))] =
              kernel_Gamma2_1loop(VELO, k3, k1, k2, qi) / (4.0 * pi);
        }
      }
    }
    save_kernel_data(base_N2, N2, nr * nx * nq * 2 * 3, ik);
  }

  return;
}

void fast_kernels::T3U3V3_kernel(void) {
  char base_T3[256], base_U3[256], base_V3[256];
  int ind;
  double ki, ri, xi, p_i, kpi, mu12, rmin, rmax, xmin, xmax, wr, wx;
  double qi, k1, k2, k3, p1, p2, p3, r1, r2, r3;
  double Pkq, Pk1, Pk2, Pk3, Pkp1, Pkp2, Pkp3, Pkr1, Pkr2, Pkr3;
  double res_T3[8], res_U3[8], res_V3[3 * 8];
  Type a, b, c;
  Vector kk1, kk2, kk3, pp1, pp2, pp3, rr1, rr2, rr3, qq;

  sprintf(base_T3, "%s/T3/T3", kernel_root.c_str());
  sprintf(base_U3, "%s/U3/U3", kernel_root.c_str());
  sprintf(base_V3, "%s/V3/V3", kernel_root.c_str());

#ifdef MPI_PARALLEL
  for (int ik = myrank; ik < nk; ik += numprocs) {
#else
  for (int ik = 0; ik < nk; ++ik) {
#endif
    ki = k[ik];
    for (int ir = 0; ir < nr; ++ir) {
      rmin = qmin / ki;
      rmax = qmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);
        kpi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p_i = ki * ri;

        k1 = p_i;
        k2 = kpi;
        k3 = ki;

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
            T3[i + 8 * (iq + nq * (ix + nx * ir))] = res_T3[i] / (4.0 * pi);
            U3[i + 8 * (iq + nq * (ix + nx * ir))] = res_U3[i] / (4.0 * pi);
          }
        }
        for (int i = 0; i < 8; ++i) {
          V3[0 + 3 * (i + 8 * (ix + nx * ir))] =
              res_V3[0 + 3 * i] / cub(2.0 * pi);
          V3[1 + 3 * (i + 8 * (ix + nx * ir))] =
              res_V3[1 + 3 * i] / cub(2.0 * pi);
          V3[2 + 3 * (i + 8 * (ix + nx * ir))] =
              res_V3[2 + 3 * i] / cub(2.0 * pi);
        }
      }
    }

    save_kernel_data(base_T3, T3, nr * nx * nq * 8, ik);
    save_kernel_data(base_U3, U3, nr * nx * nq * 8, ik);
    save_kernel_data(base_V3, V3, nr * nx * 8 * 3, ik);
  }

  return;
}
