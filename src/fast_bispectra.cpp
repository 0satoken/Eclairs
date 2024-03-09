#include "fast_bispectra.hpp"
#include "bispectra.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_spline.h>
#include <iostream>

fast_bispectra::fast_bispectra(params &params, cosmology &cosmo, spectra &spec) {
  this->para = &params;
  this->cosmo = &cosmo;
  this->spec = &spec;
  bispec = new bispectra(params, cosmo, spec);

  eta = cosmo.get_eta(); // conformal time

  c = 1.0; // boost factor
  pi = 4.0 * atan(1.0);
  nk_spl = para->iparams["nk_spl"];
  lambda = para->dparams["lambda_bispectrum"];

  /* setting parameters */
  verbose = para->bparams["verbose"];
  Kmin = para->dparams["kernel_K_Kmin"];
  Kmax = para->dparams["kernel_K_Kmax"];
  nK = para->iparams["kernel_K_nK"];
  nkb = para->iparams["fast_nkb"];
  qmin = para->dparams["fast_qmin"];
  qmax = para->dparams["fast_qmax"];
  mumin = para->dparams["fast_mumin"];
  mumax = para->dparams["fast_mumax"];
  phimin = para->dparams["fast_phimin"];
  phimax = para->dparams["fast_phimax"];

  nq = para->iparams["fast_nq"];
  nmu = para->iparams["fast_nmu"];
  nphi = para->iparams["fast_nphi"];

  fidmodels_dir = para->sparams["fast_fidmodels_dir"];
  fidmodels_config = para->sparams["fast_fidmodels_config"];
  k1min = para->dparams["fast_fidmodels_k1min"];
  k1max = para->dparams["fast_fidmodels_k1max"];
  nk1 = para->iparams["fast_fidmodels_nk1"];
  k2min = para->dparams["fast_fidmodels_k2min"];
  k2max = para->dparams["fast_fidmodels_k2max"];
  nk2 = para->iparams["fast_fidmodels_nk2"];

  flag_SPT = para->bparams["fast_SPT"];

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

  /* set up integration weights */
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

  find_nearest_fiducial();
  if(verbose) cout << "#Nearest fiducial model found" << endl;

  load_K_bin();
  if(verbose) cout << "#Loaded K bin" << endl;

  Bkfid_1l = new double[nK * nK * nK * 8];
  dBk_1l = new double[nK * nK * nK * 8];
  Qk = new double[nK * nK * nK * 8];

  set_sigmad2_spline();
  if(verbose) cout << "#setting sigmad2 spline" << endl;

  load_linear_power();
  if(verbose) cout << "#Linear power done" << endl;

  compute_fiducial_bispectra();
  if(verbose) cout << "#Fiducial bispectra done" << endl;

  compute_delta_bispectra();
  if(verbose) cout << "#delta bispectra done" << endl;
}

fast_bispectra::~fast_bispectra() {
  delete bispec;
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
  delete[] Bkfid_1l;
  delete[] dBk_1l;
  delete[] Qk;

  gsl_integration_glfixed_table_free(t_q);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_integration_glfixed_table_free(t_phi);
  gsl_spline_free(spl_fidP0);
  gsl_interp_accel_free(acc_fidP0);
  gsl_spline_free(spl_sigmad2);
  gsl_interp_accel_free(acc_sigmad2);
}

void fast_bispectra::set_sigmad2_spline(void) {
  double *logk_table, *sigmad2_table;

  logk_table = new double[nk_spl];
  sigmad2_table = new double[nk_spl];

  acc_sigmad2 = gsl_interp_accel_alloc();

  for (int i = 0; i < nk_spl; i++) {
    logk_table[i] = (log(kmax_spl) - log(kmin_spl)) / (nk_spl - 1.0) * i + log(kmin_spl);
    sigmad2_table[i] = spec->get_sigmad2(exp(logk_table[i]), lambda);
  }

  sigmad2_max = sigmad2_table[nk_spl - 1];

  spl_sigmad2 = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  gsl_spline_init(spl_sigmad2, logk_table, sigmad2_table, nk_spl);

  delete[] logk_table;
  delete[] sigmad2_table;

  return;
}

double fast_bispectra::sigmad2(double k) {
  double logk;

  logk = log(k);
  if (logk < log(kmin_spl)) {
    return 0.0;
  } else if (logk > log(kmax_spl)) {
    return sigmad2_max;
  } else {
    return gsl_spline_eval(spl_sigmad2, logk, acc_sigmad2);
  }
}

void fast_bispectra::find_nearest_fiducial(void) {
  char fname[256];
  string str;
  int nfid, fid_min, n;
  double chi2, boost, boost_chi, Ai_chi, Bi_chi, Ai, Bi;
  double chi2_min, boost_min;
  double *k_, *fidP0_, *k1, *k2, *sigma1, *sigma2, *chi2s, *boosts;
  ifstream ifs;
  vector<string> fidmodels;
  FILE *fp;
  gsl_spline *spl;
  gsl_interp_accel *acc;

  k1 = new double[nk1];
  k2 = new double[nk2];
  sigma1 = new double[nk1];
  sigma2 = new double[nk2];

  for (int i = 0; i < nk1; ++i) {
    k1[i] = (log(k1max) - log(k1min)) / ((double)nk1) * i + log(k1min);
    k1[i] = exp(k1[i]);
    sigma1[i] = k1[i];
  }

  for (int i = 0; i < nk2; ++i) {
    k2[i] = (log(k2max) - log(k2min)) / ((double)nk2) * i + log(k2min);
    k2[i] = exp(k2[i]);
    sigma2[i] = k2[i];
  }

  /* load config file */
  sprintf(fname, "%s/%s", fidmodels_dir.c_str(), fidmodels_config.c_str());
  ifs.open(fname);

  if (ifs.fail()) {
    cerr << "[ERROR] failed to load config file: " << fidmodels_config << endl;
    exit(1);
  }

  while (getline(ifs, str)) {
    if (str[0] == '#') continue;
    fidmodels.push_back(str);
  }

  ifs.close();

  nfid = fidmodels.size();

  chi2s = new double[nfid];
  boosts = new double[nfid];

  if(verbose) printf("# kernel directory:%s\n", fidmodels_dir.c_str());
  for (int i = 0; i < nfid; ++i) {
    sprintf(fname, "%s/%s/linear_power.dat", fidmodels_dir.c_str(), fidmodels[i].c_str());

    if ((fp = fopen(fname, "rb")) == NULL) {
      cerr << "[ERROR] File open error!:" << fname << endl;
      exit(1);
    }

    fread(&n, sizeof(int), 1, fp);

    k_ = new double[n];
    fidP0_ = new double[n];

    fread(k_, sizeof(double), n, fp);
    fread(fidP0_, sizeof(double), n, fp);
    fread(&n, sizeof(int), 1, fp);

    fclose(fp);

    acc = gsl_interp_accel_alloc();
    spl = gsl_spline_alloc(gsl_interp_cspline, n);
    gsl_spline_init(spl, k_, fidP0_, n);

    delete[] k_;
    delete[] fidP0_;

    Ai_chi = 0.0;
    Bi_chi = 0.0;

    for (int i1 = 0; i1 < nk1; ++i1) {
      Ai_chi += log(spec->P0(k1[i1]) / gsl_spline_eval(spl, k1[i1], acc)) /
                sqr(sigma1[i1]);
      Bi_chi += 1.0 / sqr(sigma1[i1]);
    }

    Ai = 0.0;
    Bi = 0.0;

    for (int i2 = 0; i2 < nk2; ++i2) {
      Ai += spec->P0(k2[i2]) * gsl_spline_eval(spl, k2[i2], acc) / sqr(sigma2[i2]);
      Bi += sqr(gsl_spline_eval(spl, k2[i2], acc) / sigma2[i2]);
    }

    boost_chi = sqrt(exp(Ai_chi / Bi_chi));
    boost = sqrt(Ai / Bi);

    chi2 = 0.0;
    for (int i1 = 0; i1 < nk1; ++i1) {
      chi2 += sqr(log(spec->P0(k1[i1]) /
                      (sqr(boost_chi) * gsl_spline_eval(spl, k1[i1], acc))) /
                  sigma1[i1]);
    }
    chi2 = chi2 / ((double) nk1);

    chi2s[i] = chi2;
    boosts[i] = boost;

    gsl_spline_free(spl);
    gsl_interp_accel_free(acc);
  }

  fid_min = 0;
  chi2_min = chi2s[0];
  boost_min = boosts[0];

  for (int i = 0; i < nfid; ++i) {
    if (chi2_min > chi2s[i]) {
      chi2_min = chi2s[i];
      boost_min = boosts[i];
      fid_min = i;
    }
  }

  c = sqr(boost_min);

  if(verbose){
    for (int i = 0; i < nfid; ++i) {
      printf("#%s -> chi2:%g boost:%g\n", fidmodels[i].c_str(), chi2s[i], boosts[i]);
    }
    printf("# Nearest is %s\n", fidmodels[fid_min].c_str());
    printf("# boost:%g\n", c);
  }

  kernel_root = fidmodels_dir + "/" + fidmodels[fid_min];

  delete[] k1;
  delete[] k2;
  delete[] sigma1;
  delete[] sigma2;
  delete[] chi2s;
  delete[] boosts;

  return;
}

void fast_bispectra::load_K_bin(void) {
  FILE *fp;
  int nK_s, nK_e;
  char fname[256];

  /* load K bin */
  sprintf(fname, "%s/Kbin.dat", kernel_root.c_str());

  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fread(&nK_s, sizeof(int), 1, fp);
  if (nK_s != nK) {
    cerr << "[ERROR] inconsistent header" << endl;
    exit(1);
  }

  K = new double[nK];
  fread(K, sizeof(double), nK_s, fp);
  fread(&nK_e, sizeof(int), 1, fp);

  if (nK_s != nK_e) {
    cerr << "[ERROR] inconsistent header and footer" << endl;
    exit(1);
  }

  kmin_spl = K[0];
  kmax_spl = 2.0 * K[nK - 1];

  fclose(fp);

  return;
}

void fast_bispectra::load_linear_power(void) {
  FILE *fp;
  char fname[256];
  int n;
  double *k_, *fidP0_;

  sprintf(fname, "%s/linear_power.dat", kernel_root.c_str());

  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fread(&n, sizeof(int), 1, fp);

  k_ = new double[n];
  fidP0_ = new double[n];

  fread(k_, sizeof(double), n, fp);
  fread(fidP0_, sizeof(double), n, fp);
  fread(&n, sizeof(int), 1, fp);

  fclose(fp);

  kmin_fidP0 = k_[0];
  kmax_fidP0 = k_[n - 1];

  acc_fidP0 = gsl_interp_accel_alloc();

  spl_fidP0 = gsl_spline_alloc(gsl_interp_cspline, n);
  gsl_spline_init(spl_fidP0, k_, fidP0_, n);

  delete[] k_;
  delete[] fidP0_;

  return;
}

void fast_bispectra::load_kernel(char *base, double *data, int size, int ik) {
  FILE *fp;
  char fname[256];

  sprintf(fname, "%s_K%03d.dat", base, ik);

  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fread(data, sizeof(double), size, fp);

  fclose(fp);

  return;
}

void fast_bispectra::load_kernels_bispec(int ik) {
  char base[256];

  sprintf(base, "%s/L1/L1", kernel_root.c_str());
  load_kernel(base, L1, nK * nK * nq * 2 * 3, ik);

  sprintf(base, "%s/N2/N2", kernel_root.c_str());
  load_kernel(base, N2, nK * nK * nq * 2 * 3, ik);

  sprintf(base, "%s/T3/T3", kernel_root.c_str());
  load_kernel(base, T3, nK * nK * nq * 8, ik);

  sprintf(base, "%s/U3/U3", kernel_root.c_str());
  load_kernel(base, U3, nK * nK * nq * 8, ik);

  sprintf(base, "%s/V3/V3", kernel_root.c_str());
  load_kernel(base, V3, nK * nK * 8 * 3, ik);

  return;
}

double fast_bispectra::fidP0(double k) {
  if (k < kmin_fidP0 || k > kmax_fidP0)
    return 0.0;

  return gsl_spline_eval(spl_fidP0, k, acc_fidP0);
}

void fast_bispectra::compute_fiducial_bispectra(void) {
  FILE *fp;
  char base[256], fname[256];
  int iK1_s, nK_s, iK1_e, nK_e;
  int ind;
  double K1i, k1, k2, k3, alpha_k1, alpha_k2, alpha_k3;
  double Pk1, Pk2, Pk3;
  double G1_a_k1, G1_b_k2, G1_c_k3;
  double G2_a_k2k3, G2_b_k3k1, G2_c_k1k2;
  double Bk211, Bk222, Bk321;
  double P0[3], G1_1l[6], G2_1l[6], F2[8], Bcorr222[8], Bcorr321[8];

  sprintf(base, "%s/diagram_bispec/diagram_bispec", kernel_root.c_str());

  for (int iK1 = 0; iK1 < nK; ++iK1) {
    sprintf(fname, "%s_K%03d.dat", base, iK1);

    if ((fp = fopen(fname, "rb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fread(&iK1_s, sizeof(int), 1, fp);
    fread(&nK_s, sizeof(int), 1, fp);
    fread(&K1i, sizeof(double), 1, fp);

    if (iK1_s != iK1 || nK_s != nK) {
      printf("diagram data load failed!\n");
      exit(1);
    }

    for (int iK2 = 0; iK2 < nK; ++iK2) {
      for (int iK3 = 0; iK3 < nK; ++iK3) {
        fread(P0, sizeof(double), 3, fp);
        fread(G1_1l, sizeof(double), 6, fp);
        fread(G2_1l, sizeof(double), 6, fp);
        fread(F2, sizeof(double), 6, fp);
        fread(Bcorr222, sizeof(double), 8, fp);
        fread(Bcorr321, sizeof(double), 8, fp);

        k1 = 0.5 * (K[iK2] + K[iK3]);
        k2 = 0.5 * (K[iK3] + K[iK1]);
        k3 = 0.5 * (K[iK1] + K[iK2]);

        Pk1 = P0[0];
        Pk2 = P0[1];
        Pk3 = P0[2];

        alpha_k1 = 0.5 * sqr(k1) * exp(2.0 * eta) * sigmad2(k1);
        alpha_k2 = 0.5 * sqr(k2) * exp(2.0 * eta) * sigmad2(k2);
        alpha_k3 = 0.5 * sqr(k3) * exp(2.0 * eta) * sigmad2(k3);

        for (int i1 = 0; i1 < 2; i1++) {
          for (int i2 = 0; i2 < 2; i2++) {
            for (int i3 = 0; i3 < 2; i3++) {
              ind = 4 * i1 + 2 * i2 + 1 * i3;

              if (flag_SPT) {
                /* SPT 1-loop */
                Bk211 =
                    2.0 * exp(4.0 * eta) *
                    ((F2[1 + 3 * i1] +
                      exp(2.0 * eta) * (c * G2_1l[1 + 3 * i1]) +
                      exp(2.0 * eta) * F2[1 + 3 * i1] *
                          ((c * G1_1l[1 + 3 * i2]) + (c * G1_1l[2 + 3 * i3]))) *
                         (c * Pk2) * (c * Pk3) +
                     (F2[2 + 3 * i2] +
                      exp(2.0 * eta) * (c * G2_1l[2 + 3 * i2]) +
                      exp(2.0 * eta) * F2[2 + 3 * i2] *
                          ((c * G1_1l[2 + 3 * i3]) + (c * G1_1l[0 + 3 * i1]))) *
                         (c * Pk3) * (c * Pk1) +
                     (F2[0 + 3 * i3] +
                      exp(2.0 * eta) * (c * G2_1l[0 + 3 * i3]) +
                      exp(2.0 * eta) * F2[0 + 3 * i3] *
                          ((c * G1_1l[0 + 3 * i1]) + (c * G1_1l[1 + 3 * i2]))) *
                         (c * Pk1) * (c * Pk2));
                Bk222 = exp(6.0 * eta) * (cub(c) * Bcorr222[ind]);
                Bk321 = exp(6.0 * eta) * (cub(c) * Bcorr321[ind]);

                Bkfid_1l[ind + 8 * (iK3 + nK * (iK2 + nK * iK1))] =
                    Bk211 + Bk222 + Bk321;
              } else {
                /* RegPT 1-loop */
                /*
                G1_a_k1 =
                exp(eta)*exp(-alpha_k1)*(1.0+alpha_k1+exp(2.0*eta)*(c*G1_1l[0+3*i1]));
                G1_b_k2 =
                exp(eta)*exp(-alpha_k2)*(1.0+alpha_k2+exp(2.0*eta)*(c*G1_1l[1+3*i2]));
                G1_c_k3 =
                exp(eta)*exp(-alpha_k3)*(1.0+alpha_k3+exp(2.0*eta)*(c*G1_1l[2+3*i3]));

                G2_a_k2k3 = exp(2.0*eta)*exp(-alpha_k1)*
                            ((1.0+alpha_k1)*F2[1+3*i1]+exp(2.0*eta)*(c*G2_1l[1+3*i1]));
                G2_b_k3k1 = exp(2.0*eta)*exp(-alpha_k2)*
                            ((1.0+alpha_k2)*F2[2+3*i2]+exp(2.0*eta)*(c*G2_1l[2+3*i2]));
                G2_c_k1k2 = exp(2.0*eta)*exp(-alpha_k3)*
                            ((1.0+alpha_k3)*F2[0+3*i3]+exp(2.0*eta)*(c*G2_1l[0+3*i3]));

                Bk211 = 2.0*(G2_a_k2k3*G1_b_k2*G1_c_k3*(c*Pk2)*(c*Pk3)+
                             G2_b_k3k1*G1_c_k3*G1_a_k1*(c*Pk3)*(c*Pk1)+
                             G2_c_k1k2*G1_a_k1*G1_b_k2*(c*Pk1)*(c*Pk2));
                Bk222 =
                exp(6.0*eta)*exp(-(alpha_k1+alpha_k2+alpha_k3))*(cub(c)*Bcorr222[ind]);
                Bk321 =
                exp(6.0*eta)*exp(-(alpha_k1+alpha_k2+alpha_k3))*(cub(c)*Bcorr321[ind]);

                */
                G1_a_k1 = exp(eta) * (1.0 + alpha_k1 +
                                      exp(2.0 * eta) * (c * G1_1l[0 + 3 * i1]));
                G1_b_k2 = exp(eta) * (1.0 + alpha_k2 +
                                      exp(2.0 * eta) * (c * G1_1l[1 + 3 * i2]));
                G1_c_k3 = exp(eta) * (1.0 + alpha_k3 +
                                      exp(2.0 * eta) * (c * G1_1l[2 + 3 * i3]));

                G2_a_k2k3 =
                    exp(2.0 * eta) * ((1.0 + alpha_k1) * F2[1 + 3 * i1] +
                                      exp(2.0 * eta) * (c * G2_1l[1 + 3 * i1]));
                G2_b_k3k1 =
                    exp(2.0 * eta) * ((1.0 + alpha_k2) * F2[2 + 3 * i2] +
                                      exp(2.0 * eta) * (c * G2_1l[2 + 3 * i2]));
                G2_c_k1k2 =
                    exp(2.0 * eta) * ((1.0 + alpha_k3) * F2[0 + 3 * i3] +
                                      exp(2.0 * eta) * (c * G2_1l[0 + 3 * i3]));
                Bk211 = 2.0 * exp(-(alpha_k1 + alpha_k2 + alpha_k3)) *
                        (G2_a_k2k3 * G1_b_k2 * G1_c_k3 * (c * Pk2) * (c * Pk3) +
                         G2_b_k3k1 * G1_c_k3 * G1_a_k1 * (c * Pk3) * (c * Pk1) +
                         G2_c_k1k2 * G1_a_k1 * G1_b_k2 * (c * Pk1) * (c * Pk2));
                Bk222 = exp(6.0 * eta) *
                        exp(-(alpha_k1 + alpha_k2 + alpha_k3)) *
                        (cub(c) * Bcorr222[ind]);
                Bk321 = exp(6.0 * eta) *
                        exp(-(alpha_k1 + alpha_k2 + alpha_k3)) *
                        (cub(c) * Bcorr321[ind]);

                Bkfid_1l[ind + 8 * (iK3 + nK * (iK2 + nK * iK1))] =
                    Bk211 + Bk222 + Bk321;
              }
            }
          }
        }
      }
    }

    fread(&iK1_e, sizeof(int), 1, fp);
    fread(&nK_e, sizeof(int), 1, fp);

    if (iK1_s != iK1_e || nK_s != nK_e) {
      printf("diagram data load failed!:%d\n", iK1);
      exit(1);
    }

    fclose(fp);
  }

  return;
}

void fast_bispectra::compute_delta_bispectra(void) {
  FILE *fp;
  char base[256], fname[256];
  int ind, iK1_s, nK_s, iK1_e, nK_e;
  double K1i, qi, k1, k2, k3, alpha_k1, alpha_k2, alpha_k3;
  double Pk1, Pk2, Pk3, dPk1, dPk2, dPk3, Pk1_tar, Pk2_tar, Pk3_tar;
  double Bkfid_1li, dBk_1li;
  double G1reg[6], G2reg[6], dG1_1l[6], dG2_1l[6], dG1reg[6], dG2reg[6];
  double T[8], U[8], V[8 * 3], dBcorr211_I[8], dBcorr211_II[8], dBcorr222[8],
      dBcorr321[8];
  double P0[3], G1_1l[6], G2_1l[6], F2[8], Bcorr222[8], Bcorr321[8];
  double *dP0;

  sprintf(base, "%s/diagram_bispec/diagram_bispec", kernel_root.c_str());

  dP0 = new double[nq];

  for (int iq = 0; iq < nq; ++iq) {
    qi = q[iq];
    dP0[iq] = spec->P0(qi) - (c * fidP0(qi));
  }

  for (int iK1 = 0; iK1 < nK; ++iK1) {
    sprintf(fname, "%s_K%03d.dat", base, iK1);

    if ((fp = fopen(fname, "rb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fread(&iK1_s, sizeof(int), 1, fp);
    fread(&nK_s, sizeof(int), 1, fp);
    fread(&K1i, sizeof(double), 1, fp);

    if (iK1_s != iK1 || nK_s != nK) {
      printf("diagram data load failed!:%d\n", iK1);
      exit(1);
    }

    load_kernels_bispec(iK1);

    for (int iK2 = 0; iK2 < nK; ++iK2) {
      for (int iK3 = 0; iK3 < nK; ++iK3) {
        fread(P0, sizeof(double), 3, fp);
        fread(G1_1l, sizeof(double), 6, fp);
        fread(G2_1l, sizeof(double), 6, fp);
        fread(F2, sizeof(double), 6, fp);
        fread(Bcorr222, sizeof(double), 8, fp);
        fread(Bcorr321, sizeof(double), 8, fp);

        k1 = 0.5 * (K[iK2] + K[iK3]);
        k2 = 0.5 * (K[iK3] + K[iK1]);
        k3 = 0.5 * (K[iK1] + K[iK2]);

        Pk1 = P0[0];
        Pk2 = P0[1];
        Pk3 = P0[2];

        dPk1 = spec->P0(k1) - (c * Pk1);
        dPk2 = spec->P0(k2) - (c * Pk2);
        dPk3 = spec->P0(k3) - (c * Pk3);

        alpha_k1 = 0.5 * exp(2.0 * eta) * sqr(k1) * sigmad2(k1);
        alpha_k2 = 0.5 * exp(2.0 * eta) * sqr(k2) * sigmad2(k2);
        alpha_k3 = 0.5 * exp(2.0 * eta) * sqr(k3) * sigmad2(k3);

        Pk1_tar = cosmo->Plin(k1);
        Pk2_tar = cosmo->Plin(k2);
        Pk3_tar = cosmo->Plin(k3);

        for (int i = 0; i < 6; ++i) {
          dG1_1l[i] = 0.0, dG2_1l[i] = 0.0;
        }

        for (int i = 0; i < 8; ++i) {
          T[i] = 0.0, U[i] = 0.0;
        }

        for (int iq = 0; iq < nq; ++iq) {
          qi = q[iq];

          for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
              dG1_1l[j + 3 * i] +=
                  wq[iq] * cub(qi) *
                  L1[j + 3 * (i + 2 * (iq + nq * (iK3 + nK * iK2)))] * dP0[iq];
              dG2_1l[j + 3 * i] +=
                  wq[iq] * cub(qi) *
                  N2[j + 3 * (i + 2 * (iq + nq * (iK3 + nK * iK2)))] * dP0[iq];
            }
          }

          for (int i = 0; i < 8; ++i) {
            T[i] += wq[iq] * cub(qi) *
                    (sqr(c) * T3[i + 8 * (iq + nq * (iK3 + nK * iK2))]) *
                    dP0[iq];
            U[i] += wq[iq] * cub(qi) *
                    (sqr(c) * U3[i + 8 * (iq + nq * (iK3 + nK * iK2))]) *
                    dP0[iq];
          }
        }

        for (int i = 0; i < 6; ++i) {
          dG1_1l[i] = dG1_1l[i] / (2.0 * sqr(pi));
          dG2_1l[i] = dG2_1l[i] / (2.0 * sqr(pi));
        }

        for (int i = 0; i < 8; ++i) {
          T[i] = T[i] / (2.0 * sqr(pi));
          U[i] = U[i] / (2.0 * sqr(pi));
        }

        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 3; ++j) {
            V[j + 3 * i] = sqr(c) * V3[j + 3 * (i + 8 * (iK3 + nK * iK2))];
          }
        }

        for (int i = 0; i < 2; ++i) {
          G1reg[0 + 3 * i] =
              exp(eta) * exp(-alpha_k1) *
              (1.0 + alpha_k1 + exp(2.0 * eta) * (c * G1_1l[0 + 3 * i]));
          G1reg[1 + 3 * i] =
              exp(eta) * exp(-alpha_k2) *
              (1.0 + alpha_k2 + exp(2.0 * eta) * (c * G1_1l[1 + 3 * i]));
          G1reg[2 + 3 * i] =
              exp(eta) * exp(-alpha_k3) *
              (1.0 + alpha_k3 + exp(2.0 * eta) * (c * G1_1l[2 + 3 * i]));
          G2reg[0 + 3 * i] = exp(2.0 * eta) * exp(-alpha_k3) *
                             ((1.0 + alpha_k3) * F2[0 + 3 * i] +
                              exp(2.0 * eta) * (c * G2_1l[0 + 3 * i]));
          G2reg[1 + 3 * i] = exp(2.0 * eta) * exp(-alpha_k1) *
                             ((1.0 + alpha_k1) * F2[1 + 3 * i] +
                              exp(2.0 * eta) * (c * G2_1l[1 + 3 * i]));
          G2reg[2 + 3 * i] = exp(2.0 * eta) * exp(-alpha_k2) *
                             ((1.0 + alpha_k2) * F2[2 + 3 * i] +
                              exp(2.0 * eta) * (c * G2_1l[2 + 3 * i]));
          dG1reg[0 + 3 * i] =
              exp(3.0 * eta) * exp(-alpha_k1) * dG1_1l[0 + 3 * i];
          dG1reg[1 + 3 * i] =
              exp(3.0 * eta) * exp(-alpha_k2) * dG1_1l[1 + 3 * i];
          dG1reg[2 + 3 * i] =
              exp(3.0 * eta) * exp(-alpha_k3) * dG1_1l[2 + 3 * i];
          dG2reg[0 + 3 * i] =
              exp(4.0 * eta) * exp(-alpha_k3) * dG2_1l[0 + 3 * i];
          dG2reg[1 + 3 * i] =
              exp(4.0 * eta) * exp(-alpha_k1) * dG2_1l[1 + 3 * i];
          dG2reg[2 + 3 * i] =
              exp(4.0 * eta) * exp(-alpha_k2) * dG2_1l[2 + 3 * i];
        }

        for (int i1 = 0; i1 < 2; i1++) {
          for (int i2 = 0; i2 < 2; i2++) {
            for (int i3 = 0; i3 < 2; i3++) {
              ind = 4 * i1 + 2 * i2 + 1 * i3;

              if (flag_SPT) {
                dBcorr211_I[ind] =
                    2.0 * exp(4.0 * eta) *
                    ((exp(2.0 * eta) * dG2_1l[1 + 3 * i1] +
                      exp(2.0 * eta) * F2[1 + 3 * i1] *
                          (dG1_1l[1 + 3 * i2] + dG1_1l[2 + 3 * i3])) *
                         (c * Pk2) * (c * Pk3) +
                     (F2[1 + 3 * i1] +
                      exp(2.0 * eta) * (c * G2_1l[1 + 3 * i1]) +
                      exp(2.0 * eta) * F2[1 + 3 * i1] *
                          ((c * G1_1l[1 + 3 * i2]) + (c * G1_1l[2 + 3 * i3]))) *
                         (dPk2 * (c * Pk3) + (c * Pk2) * dPk3) +
                     (exp(2.0 * eta) * dG2_1l[2 + 3 * i2] +
                      exp(2.0 * eta) * F2[2 + 3 * i2] *
                          (dG1_1l[2 + 3 * i3] + dG1_1l[0 + 3 * i1])) *
                         (c * Pk3) * (c * Pk1) +
                     (F2[2 + 3 * i2] +
                      exp(2.0 * eta) * (c * G2_1l[2 + 3 * i2]) +
                      exp(2.0 * eta) * F2[2 + 3 * i2] *
                          ((c * G1_1l[2 + 3 * i3]) + (c * G1_1l[0 + 3 * i1]))) *
                         (dPk3 * (c * Pk1) + (c * Pk3) * dPk1) +
                     (exp(2.0 * eta) * dG2_1l[0 + 3 * i3] +
                      exp(2.0 * eta) * F2[0 + 3 * i3] *
                          (dG1_1l[0 + 3 * i1] + dG1_1l[1 + 3 * i2])) *
                         (c * Pk1) * (c * Pk2) +
                     (F2[0 + 3 * i3] +
                      exp(2.0 * eta) * (c * G2_1l[0 + 3 * i3]) +
                      exp(2.0 * eta) * F2[0 + 3 * i3] *
                          ((c * G1_1l[0 + 3 * i1]) + (c * G1_1l[1 + 3 * i2]))) *
                         (dPk1 * (c * Pk2) + (c * Pk1) * dPk2));
                dBcorr211_II[ind] = 0.0;
              } else {
                dBcorr211_I[ind] =
                    2.0 * (dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] *
                               G1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                           G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] *
                               G1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                           G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] *
                               dG1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                           G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] *
                               G1reg[2 + 3 * i3] * dPk2 * (c * Pk3) +
                           G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] *
                               G1reg[2 + 3 * i3] * (c * Pk2) * dPk3 +
                           dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] *
                               G1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                           G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] *
                               G1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                           G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] *
                               dG1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                           G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] *
                               G1reg[0 + 3 * i1] * dPk3 * (c * Pk1) +
                           G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] *
                               G1reg[0 + 3 * i1] * (c * Pk3) * dPk1 +
                           dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] *
                               G1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                           G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] *
                               G1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                           G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] *
                               dG1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                           G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] *
                               G1reg[1 + 3 * i2] * dPk1 * (c * Pk2) +
                           G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] *
                               G1reg[1 + 3 * i2] * (c * Pk1) * dPk2);
                dBcorr211_II[ind] =
                    dG2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        G1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                    dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dG1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                    dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dPk2 * (c * dPk3) +
                    dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        (c * Pk2) * dPk3 +
                    dG2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        G1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                    G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        dG1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                    G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dPk2 * (c * Pk3) +
                    G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        (c * Pk2) * dPk3 +
                    dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dG1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                    G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        dG1reg[2 + 3 * i3] * (c * Pk2) * (c * Pk3) +
                    G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        dPk2 * (c * Pk3) +
                    G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        (c * Pk2) * dPk3 +
                    dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dPk2 * (c * Pk3) +
                    G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dPk2 * (c * Pk3) +
                    G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        dPk2 * (c * Pk3) +
                    G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dPk2 * dPk3 +
                    dG2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        (c * Pk2) * dPk3 +
                    G2reg[1 + 3 * i1] * dG1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        (c * Pk2) * dPk3 +
                    G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        (c * Pk2) * dPk3 +
                    G2reg[1 + 3 * i1] * G1reg[1 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dPk2 * dPk3 +
                    dG2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        G1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                    dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dG1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                    dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dPk3 * (c * Pk1) +
                    dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        (c * Pk3) * dPk1 +
                    dG2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        G1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                    G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        dG1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                    G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dPk3 * (c * Pk1) +
                    G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        (c * Pk3) * dPk1 +
                    dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] *
                        dG1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                    G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] *
                        dG1reg[0 + 3 * i1] * (c * Pk3) * (c * Pk1) +
                    G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        dPk3 * (c * Pk1) +
                    G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        (c * Pk3) * dPk1 +
                    dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dPk3 * (c * Pk1) +
                    G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dPk3 * (c * Pk1) +
                    G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        dPk3 * (c * Pk1) +
                    G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dPk3 * dPk1 +
                    dG2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        (c * Pk3) * dPk1 +
                    G2reg[2 + 3 * i2] * dG1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        (c * Pk3) * dPk1 +
                    G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        (c * Pk3) * dPk1 +
                    G2reg[2 + 3 * i2] * G1reg[2 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dPk3 * dPk1 +
                    dG2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        G1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                    dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dG1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                    dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dPk1 * (c * Pk2) +
                    dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        (c * Pk1) * dPk2 +
                    dG2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        G1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                    G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        dG1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                    G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dPk1 * (c * Pk2) +
                    G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        (c * Pk1) * dPk2 +
                    dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] *
                        dG1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                    G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] *
                        dG1reg[1 + 3 * i2] * (c * Pk1) * (c * Pk2) +
                    G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        dPk1 * (c * Pk2) +
                    G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        (c * Pk1) * dPk2 +
                    dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dPk1 * (c * Pk2) +
                    G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dPk1 * (c * Pk2) +
                    G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        dPk1 * (c * Pk2) +
                    G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dPk1 * dPk2 +
                    dG2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        (c * Pk1) * dPk2 +
                    G2reg[0 + 3 * i3] * dG1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        (c * Pk1) * dPk2 +
                    G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * dG1reg[1 + 3 * i2] *
                        (c * Pk1) * dPk2 +
                    G2reg[0 + 3 * i3] * G1reg[0 + 3 * i1] * G1reg[1 + 3 * i2] *
                        dPk1 * dPk2;
              }
            }
          }
        }

        for (int i = 0; i < 8; ++i) {
          if (flag_SPT) {
            dBcorr222[i] = 8.0 * exp(6.0 * eta) * T[i];
            dBcorr321[i] = 12.0 * exp(6.0 * eta) * U[i] +
                           6.0 * exp(6.0 * eta) *
                               (V[0 + 3 * i] * dPk1 + V[1 + 3 * i] * dPk2 +
                                V[2 + 3 * i] * dPk3);
          } else {
            dBcorr222[i] = 8.0 * exp(6.0 * eta) *
                           exp(-(alpha_k1 + alpha_k2 + alpha_k3)) * T[i];
            dBcorr321[i] = 12.0 * exp(6.0 * eta) *
                               exp(-(alpha_k1 + alpha_k2 + alpha_k3)) * U[i] +
                           6.0 * exp(6.0 * eta) *
                               exp(-(alpha_k1 + alpha_k2 + alpha_k3)) *
                               (V[0 + 3 * i] * dPk1 + V[1 + 3 * i] * dPk2 +
                                V[2 + 3 * i] * dPk3);
          }

          dBk_1li =
              dBcorr211_I[i] + dBcorr211_II[i] + dBcorr222[i] + dBcorr321[i];
          dBk_1l[i + 8 * (iK3 + nK * (iK2 + nK * iK1))] = dBk_1li;

          Bkfid_1li = Bkfid_1l[i + 8 * (iK3 + nK * (iK2 + nK * iK1))];
          Qk[i + 8 * (iK3 + nK * (iK2 + nK * iK1))] =
              (Bkfid_1li + dBk_1li) /
              (Pk1_tar * Pk2_tar + Pk2_tar * Pk3_tar + Pk3_tar * Pk1_tar);
        }
      }
    }

    fread(&iK1_e, sizeof(int), 1, fp);
    fread(&nK_e, sizeof(int), 1, fp);

    if (iK1_s != iK1_e || nK_s != nK_e) {
      printf("diagram data load failed!:%d\n", iK1);
      exit(1);
    }

    fclose(fp);
  }

  delete[] dP0;

  return;
}

double fast_bispectra::get_bispectrum(Type a, Type b, Type c, double k1,
                                      double k2, double k3) {
  int i1, i2, i3, ind, iK1, iK2, iK3;
  double res, dlogK, K1, K2, K3, Pk1, Pk2, Pk3;
  double w1, w2, w3, c000, c001, c010, c011, c100, c101, c110, c111;
  double c00, c01, c10, c11, c0, c1, Qki;

  i1 = (a == DENS) ? 0 : 1;
  i2 = (b == DENS) ? 0 : 1;
  i3 = (c == DENS) ? 0 : 1;
  ind = 4 * i1 + 2 * i2 + 1 * i3;

  dlogK = (log(Kmax) - log(Kmin)) / ((double)nK - 1);

  K1 = k2 + k3 - k1;
  K2 = k3 + k1 - k2;
  K3 = k1 + k2 - k3;

  if (K1 < 0.0 || K2 < 0.0 || K3 < 0.0) {
    return 0.0;
  }

  Pk1 = cosmo->Plin(k1);
  Pk2 = cosmo->Plin(k2);
  Pk3 = cosmo->Plin(k3);

  iK1 = (int)floor((log(K1) - log(Kmin)) / dlogK);
  iK2 = (int)floor((log(K2) - log(Kmin)) / dlogK);
  iK3 = (int)floor((log(K3) - log(Kmin)) / dlogK);

  if (iK1 >= nK - 1 || iK2 >= nK - 1 || iK3 >= nK - 1) {
    res = 0.0;
  } else if (iK1 < 0 || iK2 < 0 || iK3 < 0) {
    res = bispec->Bispec_tree(a, b, c, k1, k2, k3);
  } else {
    w1 = (log(K1) - log(Kmin)) / dlogK - double(iK1);
    w2 = (log(K2) - log(Kmin)) / dlogK - double(iK2);
    w3 = (log(K3) - log(Kmin)) / dlogK - double(iK3);

    c000 = Qk[ind + 8 * (iK3 + nK * (iK2 + nK * iK1))];
    c001 = Qk[ind + 8 * ((iK3 + 1) + nK * (iK2 + nK * iK1))];
    c010 = Qk[ind + 8 * (iK3 + nK * ((iK2 + 1) + nK * iK1))];
    c011 = Qk[ind + 8 * ((iK3 + 1) + nK * ((iK2 + 1) + nK * iK1))];
    c100 = Qk[ind + 8 * (iK3 + nK * (iK2 + nK * (iK1 + 1)))];
    c101 = Qk[ind + 8 * ((iK3 + 1) + nK * (iK2 + nK * (iK1 + 1)))];
    c110 = Qk[ind + 8 * (iK3 + nK * ((iK2 + 1) + nK * (iK1 + 1)))];
    c111 = Qk[ind + 8 * ((iK3 + 1) + nK * ((iK2 + 1) + nK * (iK1 + 1)))];

    c00 = c000 * (1.0 - w1) + c100 * w1;
    c01 = c001 * (1.0 - w1) + c101 * w1;
    c10 = c010 * (1.0 - w1) + c110 * w1;
    c11 = c011 * (1.0 - w1) + c111 * w1;

    c0 = c00 * (1.0 - w2) + c10 * w2;
    c1 = c01 * (1.0 - w2) + c11 * w2;

    Qki = c0 * (1.0 - w3) + c1 * w3;

    res = Qki * (Pk1 * Pk2 + Pk2 * Pk3 + Pk3 * Pk1);
  }

  return res;
}

vector<double>
fast_bispectra::get_binned_bispectrum(Type a, Type b, Type c,
                                      map<string, vector<double>> kbin) {
  int nkbin;
  double kb1min, kb1max, deltakb1, kb2min, kb2max, deltakb2, kb3min, kb3max,
      deltakb3;
  double Bki, binnedBki, Nk, Vk;
  double k1, k2, k3;
  vector<double> res;

  nkbin = kbin["k1min"].size();
  res.resize(nkbin);

  for (int i = 0; i < nkbin; ++i) {
    kb1min = kbin["k1min"][i];
    kb1max = kbin["k1max"][i];
    deltakb1 = (kb1max - kb1min) / nkb;

    kb2min = kbin["k2min"][i];
    kb2max = kbin["k2max"][i];
    deltakb2 = (kb2max - kb2min) / nkb;

    kb3min = kbin["k3min"][i];
    kb3max = kbin["k3max"][i];
    deltakb3 = (kb3max - kb3min) / nkb;

    binnedBki = 0.0;
    Nk = 0.0;

    for (int i1 = 0; i1 < nkb; ++i1) {
      k1 = deltakb1 * (i1 + 0.5) + kb1min;
      for (int i2 = 0; i2 < nkb; ++i2) {
        k2 = deltakb2 * (i2 + 0.5) + kb2min;
        for (int i3 = 0; i3 < nkb; ++i3) {
          k3 = deltakb3 * (i3 + 0.5) + kb3min;

          if (k1 > k2 + k3 || k2 > k3 + k1 || k3 > k1 + k2)
            continue;

          Vk = k1 * k2 * k3 * deltakb1 * deltakb2 * deltakb3;

          Bki = get_bispectrum(a, b, c, k1, k2, k3);
          binnedBki += Vk * Bki;
          Nk += Vk;
        }
      }
    }

    if (Nk > 0.0) {
      binnedBki /= Nk;
      res[i] = binnedBki;
    } else {
      res[i] = 0.0;
    }
  }

  return res;
}
