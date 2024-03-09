#include "fast_spectra.hpp"
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

fast_spectra::fast_spectra(params &params, cosmology &cosmo, spectra &spec) {
  this->para = &params;
  this->cosmo = &cosmo;
  this->spec = &spec;

  f = cosmo.get_growth_rate(); // growth rate
  eta = cosmo.get_eta();       // conformal time

  c = 1.0; // boost factor
  pi = 4.0 * atan(1.0);
  nk_spl = para->iparams["nk_spl"];

  /* bias parameter */
  b1 = para->dparams["b1"];
  beta = f / b1;

  /* setting parameters */
  verbose = para->bparams["verbose"];
  kmin = para->dparams["kmin"];
  kmax = para->dparams["kmax"];
  lambda_p = para->dparams["lambda_power"];
  lambda_b = para->dparams["lambda_bispectrum"];
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

  direct_Bterm = para->bparams["fast_direct_Bterm"];
  flag_SPT = para->bparams["fast_SPT"];

  fidmodels_dir = para->sparams["fast_fidmodels_dir"];
  fidmodels_config = para->sparams["fast_fidmodels_config"];
  k1min = para->dparams["fast_fidmodels_k1min"];
  k1max = para->dparams["fast_fidmodels_k1max"];
  nk1 = para->iparams["fast_fidmodels_nk1"];
  k2min = para->dparams["fast_fidmodels_k2min"];
  k2max = para->dparams["fast_fidmodels_k2max"];
  nk2 = para->iparams["fast_fidmodels_nk2"];

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

  acc_Pk_1l = new gsl_interp_accel *[3];
  acc_Pk_2l = new gsl_interp_accel *[3];
  acc_A = new gsl_interp_accel *[3];
  acc_B = new gsl_interp_accel *[4];

  spl_Pk_1l = new gsl_spline *[3];
  spl_Pk_2l = new gsl_spline *[3];
  spl_A = new gsl_spline *[3];
  spl_B = new gsl_spline *[4];

  /* set up integration weights */
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

  find_nearest_fiducial();
  if(verbose) cout << "# Nearest fiducial model found" << endl;

  load_k_bin();
  if(verbose) cout << "# Loaded k bin" << endl;

  Pkfid_1l = new double[nk * 3];
  Pkfid_2l = new double[nk * 3];
  Bkfid_1l = new double[nk * nr * nx * 8];

  dPk_1l = new double[nk * 3];
  dPk_2l = new double[nk * 3];
  dBk_1l = new double[nk * nr * nx * 8];

  A = new double[nk * 3];
  B = new double[nk * 4];

  set_sigmad2_spline();
  if(verbose) cout << "# setting sigmad2 spline" << endl;

  load_linear_power();
  if(verbose) cout << "# Linear power done" << endl;

  compute_fiducial_spectra();
  if(verbose) cout << "# Fiducial spectra done" << endl;

  compute_fiducial_bispectra();
  if(verbose) cout << "# Fiducial bispectra done" << endl;

  compute_delta_spectra();
  if(verbose) cout << "# delta spectra done" << endl;

  compute_delta_bispectra();
  if(verbose) cout << "# delta bispectra done" << endl;

  construct_spline_spectra();

  Aterm_recon();
  construct_spline_Aterm();
  if(verbose) cout << "#A-term done" << endl;

  Bterm_recon();
  construct_spline_Bterm();
  if(verbose) cout << "#B-term done" << endl;
}

fast_spectra::~fast_spectra() {
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
  delete[] Pkfid_1l;
  delete[] Pkfid_2l;
  delete[] Bkfid_1l;
  delete[] dPk_1l;
  delete[] dPk_2l;
  delete[] dBk_1l;
  delete[] A;
  delete[] B;

  gsl_integration_glfixed_table_free(t_q);
  gsl_integration_glfixed_table_free(t_p);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_integration_glfixed_table_free(t_phi);
  gsl_integration_glfixed_table_free(t_r);
  gsl_integration_glfixed_table_free(t_x);
  gsl_spline_free(spl_fidP0);
  gsl_interp_accel_free(acc_fidP0);
  gsl_spline_free(spl_sigmad2_p);
  gsl_interp_accel_free(acc_sigmad2_p);
  gsl_spline_free(spl_sigmad2_b);
  gsl_interp_accel_free(acc_sigmad2_b);

  for (int i = 0; i < 3; ++i) {
    gsl_spline_free(spl_Pk_1l[i]);
    gsl_spline_free(spl_Pk_2l[i]);
    gsl_interp_accel_free(acc_Pk_1l[i]);
    gsl_interp_accel_free(acc_Pk_2l[i]);
  }

  for (int i = 0; i < 3; ++i) {
    gsl_spline_free(spl_A[i]);
    gsl_interp_accel_free(acc_A[i]);
  }

  for (int i = 0; i < 4; ++i) {
    gsl_spline_free(spl_B[i]);
    gsl_interp_accel_free(acc_B[i]);
  }

  delete[] spl_Pk_1l;
  delete[] spl_Pk_2l;
  delete[] spl_A;
  delete[] spl_B;
  delete[] acc_Pk_1l;
  delete[] acc_Pk_2l;
  delete[] acc_A;
  delete[] acc_B;
}

void fast_spectra::set_sigmad2_spline(void) {
  double *logk_table, *sigmad2_p_table, *sigmad2_b_table;

  logk_table = new double[nk_spl];
  sigmad2_p_table = new double[nk_spl];
  sigmad2_b_table = new double[nk_spl];

  acc_sigmad2_p = gsl_interp_accel_alloc();
  acc_sigmad2_b = gsl_interp_accel_alloc();

  for (int i = 0; i < nk_spl; i++) {
    logk_table[i] = (log(kmax) - log(kmin)) / (nk_spl - 1.0) * i + log(kmin);
    sigmad2_p_table[i] = spec->get_sigmad2(exp(logk_table[i]), lambda_p);
    sigmad2_b_table[i] = spec->get_sigmad2(exp(logk_table[i]), lambda_b);
  }

  sigmad2_p_max = sigmad2_p_table[nk_spl - 1];
  sigmad2_b_max = sigmad2_b_table[nk_spl - 1];

  spl_sigmad2_p = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  spl_sigmad2_b = gsl_spline_alloc(gsl_interp_cspline, nk_spl);

  gsl_spline_init(spl_sigmad2_p, logk_table, sigmad2_p_table, nk_spl);
  gsl_spline_init(spl_sigmad2_b, logk_table, sigmad2_b_table, nk_spl);

  delete[] logk_table;
  delete[] sigmad2_p_table;
  delete[] sigmad2_b_table;

  return;
}

double fast_spectra::sigmad2_p(double k) {
  double logk;

  logk = log(k);
  if (logk < log(kmin)) {
    return 0.0;
  } else if (logk > log(kmax)) {
    return sigmad2_p_max;
  } else {
    return gsl_spline_eval(spl_sigmad2_p, logk, acc_sigmad2_p);
  }
}

double fast_spectra::sigmad2_b(double k) {
  double logk;

  logk = log(k);
  if (logk < log(kmin)) {
    return 0.0;
  } else if (logk > log(kmax)) {
    return sigmad2_b_max;
  } else {
    return gsl_spline_eval(spl_sigmad2_b, logk, acc_sigmad2_b);
  }
}

void fast_spectra::find_nearest_fiducial(void) {
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
    if (str[0] == '#')
      continue;
    fidmodels.push_back(str);
  }

  ifs.close();

  nfid = fidmodels.size();

  chi2s = new double[nfid];
  boosts = new double[nfid];

  printf("# kernel directory:%s\n", fidmodels_dir.c_str());
  for (int i = 0; i < nfid; ++i) {
    sprintf(fname, "%s/%s/linear_power.dat", fidmodels_dir.c_str(),
            fidmodels[i].c_str());

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
      chi2 += sqr(log(spec->P0(k1[i1]) / (sqr(boost_chi) * gsl_spline_eval(spl, k1[i1], acc))) / sigma1[i1]);
    }
    chi2 = chi2 / ((double)nk1);

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

void fast_spectra::load_k_bin(void) {
  FILE *fp;
  int nk_s, nk_e;
  char fname[256];

  /* load k bin */
  sprintf(fname, "%s/kbin.dat", kernel_root.c_str());

  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fread(&nk_s, sizeof(int), 1, fp);
  k = new double[nk_s];
  fread(k, sizeof(double), nk_s, fp);
  fread(&nk_e, sizeof(int), 1, fp);

  if (nk_s != nk_e) {
    cerr << "[ERROR] inconsistent header and footer" << endl;
    exit(1);
  }

  nk = nk_s;
  kmin_spl = k[0];
  kmax_spl = k[nk-1];


  fclose(fp);

  return;
}

void fast_spectra::load_linear_power(void) {
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

void fast_spectra::load_kernel(char *base, double *data, int size, int ik) {
  FILE *fp;
  char fname[256];

  sprintf(fname, "%s_k%03d.dat", base, ik);

  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("File open error!:%s\n", fname);
    exit(1);
  }

  fread(data, sizeof(double), size, fp);

  fclose(fp);

  return;
}

void fast_spectra::load_kernels_spec(int ik) {
  char base[256];

  sprintf(base, "%s/L1/L1", kernel_root.c_str());
  load_kernel(base, L1, nq * 2 + nr * nx * nq * 2 * 3, ik);

  sprintf(base, "%s/M1/M1", kernel_root.c_str());
  load_kernel(base, M1, nq * 2, ik);

  sprintf(base, "%s/X2/X2", kernel_root.c_str());
  load_kernel(base, X2, nq * 4, ik);

  sprintf(base, "%s/Y2/Y2", kernel_root.c_str());
  load_kernel(base, Y2, nq * 4, ik);

  sprintf(base, "%s/Z2/Z2", kernel_root.c_str());
  load_kernel(base, Z2, nq * 4, ik);

  sprintf(base, "%s/Q2/Q2", kernel_root.c_str());
  load_kernel(base, Q2, nq * 4, ik);

  sprintf(base, "%s/R2/R2", kernel_root.c_str());
  load_kernel(base, R2, nq * 4, ik);

  sprintf(base, "%s/S3/S3", kernel_root.c_str());
  load_kernel(base, S3, nq * 4, ik);

  return;
}

void fast_spectra::load_kernels_bispec(int ik) {
  char base[256];

  sprintf(base, "%s/L1/L1", kernel_root.c_str());
  load_kernel(base, L1, nq * 2 + nr * nx * nq * 2 * 3, ik);

  sprintf(base, "%s/N2/N2", kernel_root.c_str());
  load_kernel(base, N2, nr * nx * nq * 2 * 3, ik);

  sprintf(base, "%s/T3/T3", kernel_root.c_str());
  load_kernel(base, T3, nr * nx * nq * 8, ik);

  sprintf(base, "%s/U3/U3", kernel_root.c_str());
  load_kernel(base, U3, nr * nx * nq * 8, ik);

  sprintf(base, "%s/V3/V3", kernel_root.c_str());
  load_kernel(base, V3, nr * nx * 8 * 3, ik);

  return;
}

double fast_spectra::fidP0(double k) {
  if (k < kmin_fidP0 || k > kmax_fidP0)
    return 0.0;

  return gsl_spline_eval(spl_fidP0, k, acc_fidP0);
}

void fast_spectra::compute_fiducial_spectra(void) {
  FILE *fp;
  char base[256], fname[256];
  int ind, ik_s, nk_s, ik_e, nk_e;
  double ki, P0i;
  double G1_1l[2], G1_2l[2];
  double Pcorr2_00[3], Pcorr2_01[3], Pcorr2_11[3], Pcorr3[3];
  double Preg1_1l, Preg2_1l, Preg1_2l, Preg2_2l, Preg3_2l;
  double alpha;

  sprintf(base, "%s/diagram_spec/diagram_spec", kernel_root.c_str());

  for (int ik = 0; ik < nk; ++ik) {
    sprintf(fname, "%s_k%03d.dat", base, ik);

    if ((fp = fopen(fname, "rb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fread(&ik_s, sizeof(int), 1, fp);
    fread(&nk_s, sizeof(int), 1, fp);

    if (ik_s != ik || nk_s != nk) {
      printf("diagram data load failed!\n");
      exit(1);
    }

    fread(&ki, sizeof(double), 1, fp);
    fread(&P0i, sizeof(double), 1, fp);

    fread(G1_1l, sizeof(double), 2, fp);
    fread(G1_2l, sizeof(double), 2, fp);

    fread(Pcorr2_00, sizeof(double), 3, fp);
    fread(Pcorr2_01, sizeof(double), 3, fp);
    fread(Pcorr2_11, sizeof(double), 3, fp);
    fread(Pcorr3, sizeof(double), 3, fp);

    fread(&ik_e, sizeof(int), 1, fp);
    fread(&nk_e, sizeof(int), 1, fp);

    if (ik_s != ik_e || nk_s != nk_e) {
      printf("diagram data load failed!:%d\n", ik);
      exit(1);
    }

    fclose(fp);

    alpha = 0.5 * sqr(ki) * exp(2.0 * eta) * sigmad2_p(ki);

    for (int i1 = 0; i1 < 2; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        if (i1 == 0 && i2 == 0)
          ind = 0;
        else if (i1 == 0 && i2 == 1)
          ind = 1;
        else if (i1 == 1 && i2 == 0)
          continue; // Cross-spectrum is computed once.
        else if (i1 == 1 && i2 == 1)
          ind = 2;

        if (flag_SPT) {
          /* SPT 1-loop */
          Pkfid_1l[ind + 3 * ik] =
              exp(2.0 * eta) * (c * P0i) +
              exp(4.0 * eta) * (c * G1_1l[i1] + c * G1_1l[i2]) * (c * P0i) +
              exp(4.0 * eta) * sqr(c) * Pcorr2_00[ind];

          /* SPT 2-loop */
          Pkfid_2l[ind + 3 * ik] =
              exp(2.0 * eta) * (c * P0i) +
              exp(4.0 * eta) * (c * G1_1l[i1] + c * G1_1l[i2]) * (c * P0i) +
              exp(4.0 * eta) * sqr(c) * Pcorr2_00[ind] +
              exp(6.0 * eta) * (c * G1_1l[i1]) * (c * G1_1l[i2]) * (c * P0i) +
              exp(6.0 * eta) * (sqr(c) * G1_2l[i1] + sqr(c) * G1_2l[i2]) *
                  (c * P0i) +
              exp(6.0 * eta) * cub(c) * Pcorr2_01[ind] +
              exp(6.0 * eta) * cub(c) * Pcorr3[ind];
        } else {
          /* RegPT 1-loop */
          Preg1_1l = exp(2.0 * eta) * exp(-2.0 * alpha) *
                     (1.0 + alpha + exp(2.0 * eta) * (c * G1_1l[i1])) *
                     (1.0 + alpha + exp(2.0 * eta) * (c * G1_1l[i2])) *
                     (c * P0i);

          Preg2_1l =
              exp(4.0 * eta) * exp(-2.0 * alpha) * (sqr(c) * Pcorr2_00[ind]);

          Pkfid_1l[ind + 3 * ik] = Preg1_1l + Preg2_1l;

          /* RegPT 2-loop */
          Preg1_2l = exp(2.0 * eta) * exp(-2.0 * alpha) *
                     (1.0 + alpha + 0.5 * sqr(alpha) +
                      exp(2.0 * eta) * (c * G1_1l[i1]) * (1.0 + alpha) +
                      exp(4.0 * eta) * (sqr(c) * G1_2l[i1])) *
                     (1.0 + alpha + 0.5 * sqr(alpha) +
                      exp(2.0 * eta) * (c * G1_1l[i2]) * (1.0 + alpha) +
                      exp(4.0 * eta) * (sqr(c) * G1_2l[i2])) *
                     (c * P0i);

          Preg2_2l =
              exp(4.0 * eta) * exp(-2.0 * alpha) *
              ((sqr(c) * Pcorr2_00[ind]) * sqr(1.0 + alpha) +
               (cub(c) * Pcorr2_01[ind]) * exp(2.0 * eta) * (1.0 + alpha) +
               (qua(c) * Pcorr2_11[ind]) * exp(4.0 * eta));

          Preg3_2l =
              exp(6.0 * eta) * exp(-2.0 * alpha) * (cub(c) * Pcorr3[ind]);

          Pkfid_2l[ind + 3 * ik] = Preg1_2l + Preg2_2l + Preg3_2l;
        }
      }
    }
  }

  return;
}

void fast_spectra::compute_fiducial_bispectra(void) {
  FILE *fp;
  char base[256], fname[256];
  int ik_s, nk_s, nr_s, nx_s, ik_e, nk_e, nr_e, nx_e;
  int ind;
  double ki, ri, xi, kpi, p_i, k1, k2, k3, alpha_k1, alpha_k2, alpha_k3;
  double Pk1, Pk2, Pk3;
  double G1_a_k1, G1_b_k2, G1_c_k3;
  double G2_a_k2k3, G2_b_k3k1, G2_c_k1k2;
  double Bk211, Bk222, Bk321;
  double P0[3], G1_1l[6], G2_1l[6], F2[8], Bcorr222[8], Bcorr321[8];

  sprintf(base, "%s/diagram_bispec/diagram_bispec", kernel_root.c_str());

  for (int ik = 0; ik < nk; ++ik) {
    sprintf(fname, "%s_k%03d.dat", base, ik);

    if ((fp = fopen(fname, "rb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fread(&ik_s, sizeof(int), 1, fp);
    fread(&nk_s, sizeof(int), 1, fp);
    fread(&nr_s, sizeof(int), 1, fp);
    fread(&nx_s, sizeof(int), 1, fp);
    fread(&ki, sizeof(double), 1, fp);

    if (ik_s != ik || nk_s != nk || nr_s != nr || nx_s != nx) {
      printf("diagram data load failed!\n");
      exit(1);
    }

    for (int ir = 0; ir < nr; ++ir) {
      for (int ix = 0; ix < nx; ++ix) {
        fread(&ri, sizeof(double), 1, fp);
        fread(&xi, sizeof(double), 1, fp);
        fread(P0, sizeof(double), 3, fp);
        fread(G1_1l, sizeof(double), 6, fp);
        fread(G2_1l, sizeof(double), 6, fp);
        fread(F2, sizeof(double), 6, fp);
        fread(Bcorr222, sizeof(double), 8, fp);
        fread(Bcorr321, sizeof(double), 8, fp);

        kpi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p_i = ki * ri;

        k1 = p_i;
        k2 = kpi;
        k3 = ki;

        Pk1 = P0[0];
        Pk2 = P0[1];
        Pk3 = P0[2];

        alpha_k1 = 0.5 * sqr(k1) * exp(2.0 * eta) * sigmad2_b(k1);
        alpha_k2 = 0.5 * sqr(k2) * exp(2.0 * eta) * sigmad2_b(k2);
        alpha_k3 = 0.5 * sqr(k3) * exp(2.0 * eta) * sigmad2_b(k3);

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

                Bkfid_1l[ind + 8 * (ix + nx * (ir + nr * ik))] =
                    Bk211 + Bk222 + Bk321;
              } else {
                /* RegPT 1-loop */
                G1_a_k1 =
                    exp(eta) * exp(-alpha_k1) *
                    (1.0 + alpha_k1 + exp(2.0 * eta) * (c * G1_1l[0 + 3 * i1]));
                G1_b_k2 =
                    exp(eta) * exp(-alpha_k2) *
                    (1.0 + alpha_k2 + exp(2.0 * eta) * (c * G1_1l[1 + 3 * i2]));
                G1_c_k3 =
                    exp(eta) * exp(-alpha_k3) *
                    (1.0 + alpha_k3 + exp(2.0 * eta) * (c * G1_1l[2 + 3 * i3]));

                G2_a_k2k3 = exp(2.0 * eta) * exp(-alpha_k1) *
                            ((1.0 + alpha_k1) * F2[1 + 3 * i1] +
                             exp(2.0 * eta) * (c * G2_1l[1 + 3 * i1]));
                G2_b_k3k1 = exp(2.0 * eta) * exp(-alpha_k2) *
                            ((1.0 + alpha_k2) * F2[2 + 3 * i2] +
                             exp(2.0 * eta) * (c * G2_1l[2 + 3 * i2]));
                G2_c_k1k2 = exp(2.0 * eta) * exp(-alpha_k3) *
                            ((1.0 + alpha_k3) * F2[0 + 3 * i3] +
                             exp(2.0 * eta) * (c * G2_1l[0 + 3 * i3]));

                Bk211 = 2.0 *
                        (G2_a_k2k3 * G1_b_k2 * G1_c_k3 * (c * Pk2) * (c * Pk3) +
                         G2_b_k3k1 * G1_c_k3 * G1_a_k1 * (c * Pk3) * (c * Pk1) +
                         G2_c_k1k2 * G1_a_k1 * G1_b_k2 * (c * Pk1) * (c * Pk2));
                Bk222 = exp(6.0 * eta) *
                        exp(-(alpha_k1 + alpha_k2 + alpha_k3)) *
                        (cub(c) * Bcorr222[ind]);
                Bk321 = exp(6.0 * eta) *
                        exp(-(alpha_k1 + alpha_k2 + alpha_k3)) *
                        (cub(c) * Bcorr321[ind]);

                Bkfid_1l[ind + 8 * (ix + nx * (ir + nr * ik))] =
                    Bk211 + Bk222 + Bk321;
              }
            }
          }
        }
      }
    }

    fread(&ik_e, sizeof(int), 1, fp);
    fread(&nk_e, sizeof(int), 1, fp);
    fread(&nr_e, sizeof(int), 1, fp);
    fread(&nx_e, sizeof(int), 1, fp);

    if (ik_s != ik_e || nk_s != nk_e || nr_s != nr_e || nx_s != nx_e) {
      printf("diagram data load failed!:%d\n", ik);
      exit(1);
    }

    fclose(fp);
  }

  return;
}

void fast_spectra::compute_delta_spectra(void) {
  FILE *fp;
  char base[256], fname[256];
  int ik_s, ik_e, nk_s, nk_e, ind;
  double ki, qi, P0i, dP0i, alpha;
  double G1_1l[2], G1_2l[2], Pcorr2_00[3], Pcorr2_01[3], Pcorr2_11[3], Pcorr3[3];
  double dG1_1l[2], dG1_2l[2], X[4], Y[4], Z[4], Q[4], R[4], S[4];
  double G1reg_2l[2], G2reg_2l[2], G1reg_1l[2], G2reg_1l[2];
  double dG1reg_2l[2], dG1reg_1l[2];
  double *dP0;

  sprintf(base, "%s/diagram_spec/diagram_spec", kernel_root.c_str());

  dP0 = new double[nq];

  for (int iq = 0; iq < nq; ++iq) {
    qi = q[iq];
    dP0[iq] = spec->P0(qi) - (c * fidP0(qi));
  }

  for (int ik = 0; ik < nk; ++ik) {
    sprintf(fname, "%s_k%03d.dat", base, ik);

    if ((fp = fopen(fname, "rb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fread(&ik_s, sizeof(int), 1, fp);
    fread(&nk_s, sizeof(int), 1, fp);

    if (ik_s != ik || nk_s != nk) {
      printf("diagram data load failed!:%d\n", ik);
      exit(1);
    }

    fread(&ki, sizeof(double), 1, fp);
    fread(&P0i, sizeof(double), 1, fp);

    fread(G1_1l, sizeof(double), 2, fp);
    fread(G1_2l, sizeof(double), 2, fp);
    fread(Pcorr2_00, sizeof(double), 3, fp);
    fread(Pcorr2_01, sizeof(double), 3, fp);
    fread(Pcorr2_11, sizeof(double), 3, fp);
    fread(Pcorr3, sizeof(double), 3, fp);

    fread(&ik_e, sizeof(int), 1, fp);
    fread(&nk_e, sizeof(int), 1, fp);

    if (ik_s != ik_e || nk_s != nk_e) {
      printf("diagram data load failed!:%d\n", ik);
      exit(1);
    }

    fclose(fp);

    load_kernels_spec(ik);

    dP0i = spec->P0(ki) - (c * P0i);
    alpha = 0.5 * sqr(ki) * exp(2.0 * eta) * sigmad2_p(ki);

    for (int i = 0; i < 2; ++i) {
      G1reg_1l[i] = exp(eta) * exp(-alpha) *
                    (1.0 + alpha + exp(2.0 * eta) * (c * G1_1l[i]));
      G1reg_2l[i] = exp(eta) * exp(-alpha) *
                    (1.0 + alpha + 0.5 * sqr(alpha) +
                     exp(2.0 * eta) * (1.0 + alpha) * (c * G1_1l[i]) +
                     exp(4.0 * eta) * (sqr(c) * G1_2l[i]));
    }

    for (int i = 0; i < 2; ++i) {
      dG1_1l[i] = 0.0;
      dG1_2l[i] = 0.0;
    }

    for (int i = 0; i < 4; ++i) {
      X[i] = 0.0, Y[i] = 0.0, Z[i] = 0.0;
      Q[i] = 0.0, R[i] = 0.0, S[i] = 0.0;
    }

    for (int iq = 0; iq < nq; ++iq) {
      qi = q[iq];
      for (int i = 0; i < 2; ++i) {
        dG1_1l[i] += wq[iq] * cub(qi) * L1[i + 2 * iq] * dP0[iq];
        dG1_2l[i] += wq[iq] * cub(qi) * (c * M1[i + 2 * iq]) * dP0[iq];
      }

      for (int i = 0; i < 4; ++i) {
        X[i] += wq[iq] * cub(qi) * (c * X2[i + 4 * iq]) * dP0[iq];
        Y[i] += wq[iq] * cub(qi) * (sqr(c) * Y2[i + 4 * iq]) * dP0[iq];
        Z[i] += wq[iq] * cub(qi) * (cub(c) * Z2[i + 4 * iq]) * dP0[iq];
        Q[i] += wq[iq] * cub(qi) * (sqr(c) * Q2[i + 4 * iq]) * dP0[iq];
        R[i] += wq[iq] * cub(qi) * (cub(c) * R2[i + 4 * iq]) * dP0[iq];
        S[i] += wq[iq] * cub(qi) * (sqr(c) * S3[i + 4 * iq]) * dP0[iq];
      }
    }

    for (int i = 0; i < 2; ++i) {
      dG1_1l[i] = dG1_1l[i] / (2.0 * sqr(pi));
      dG1_2l[i] = dG1_2l[i] * 2.0 / (2.0 * sqr(pi));
    }

    for (int i = 0; i < 4; ++i) {
      X[i] = X[i] / (2.0 * sqr(pi));
      Y[i] = Y[i] / (2.0 * sqr(pi));
      Z[i] = Z[i] / (2.0 * sqr(pi));
      Q[i] = Q[i] / (2.0 * sqr(pi));
      R[i] = R[i] / (2.0 * sqr(pi));
      S[i] = S[i] / (2.0 * sqr(pi));
    }

    for (int i = 0; i < 2; ++i) {
      dG1reg_2l[i] = exp(3.0 * eta) * exp(-alpha) *
                     ((1.0 + alpha) * dG1_1l[i] + exp(2.0 * eta) * dG1_2l[i]);
      dG1reg_1l[i] = exp(3.0 * eta) * exp(-alpha) * dG1_1l[i];
    }

    for (int i1 = 0; i1 < 2; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        if (i1 == 0 && i2 == 0)
          ind = 0;
        else if (i1 == 0 && i2 == 1)
          ind = 1;
        else if (i1 == 1 && i2 == 0)
          continue; // Cross-spectrum is computed once.
        else if (i1 == 1 && i2 == 1)
          ind = 2;

        if (flag_SPT) {
          dPk_1l[ind + 3 * ik] =
              exp(2.0 * eta) * dP0i +
              exp(4.0 * eta) * (G1_1l[i1] + G1_1l[i2]) * dP0i +
              exp(4.0 * eta) * (dG1_1l[i1] + dG1_1l[i2]) * (c * P0i) +
              4.0 * exp(4.0 * eta) * X[i2 + 2 * i1];

          dPk_2l[ind + 3 * ik] =
              exp(2.0 * eta) * dP0i +
              exp(4.0 * eta) * (G1_1l[i1] + G1_1l[i2]) * dP0i +
              exp(4.0 * eta) * (dG1_1l[i1] + dG1_1l[i2]) * (c * P0i) +
              exp(6.0 * eta) *
                  (dG1_1l[i1] * G1_1l[i2] + G1_1l[i1] * dG1_1l[i2]) *
                  (c * P0i) +
              exp(6.0 * eta) * G1_1l[i1] * G1_1l[i2] * dP0i +
              exp(6.0 * eta) * (dG1_2l[i1] + dG1_2l[i2]) * (c * P0i) +
              exp(6.0 * eta) * (G1_2l[i1] + G1_2l[i2]) * dP0i +
              4.0 * exp(4.0 * eta) *
                  (X[i2 + 2 * i1] +
                   exp(2.0 * eta) * (Y[i2 + 2 * i1] + Y[i1 + 2 * i2])) +
              2.0 * exp(6.0 * eta) * (Q[i2 + 2 * i1] + Q[i1 + 2 * i2]) +
              18.0 * exp(6.0 * eta) * S[i2 + 2 * i1];
        } else {
          dPk_1l[ind + 3 * ik] =
              dG1reg_1l[i1] * G1reg_1l[i2] * (c * P0i) +
              G1reg_1l[i1] * dG1reg_1l[i2] * (c * P0i) +
              G1reg_1l[i1] * G1reg_1l[i2] * dP0i +
              4.0 * exp(4.0 * eta) * exp(-2.0 * alpha) * X[i2 + 2 * i1];

          dPk_2l[ind + 3 * ik] =
              dG1reg_2l[i1] * G1reg_2l[i2] * (c * P0i) +
              G1reg_2l[i1] * dG1reg_2l[i2] * (c * P0i) +
              G1reg_2l[i1] * G1reg_2l[i2] * dP0i +
              4.0 * exp(4.0 * eta) * exp(-2.0 * alpha) *
                  (sqr(1.0 + alpha) * X[i2 + 2 * i1] +
                   exp(2.0 * eta) * (1.0 + alpha) *
                       (Y[i2 + 2 * i1] + Y[i1 + 2 * i2]) +
                   exp(4.0 * eta) * Z[i2 + 2 * i1]) +
              2.0 * exp(6.0 * eta) * exp(-2.0 * alpha) *
                  ((1.0 + alpha) * (Q[i2 + 2 * i1] + Q[i1 + 2 * i2]) +
                   exp(2.0 * eta) * (R[i2 + 2 * i1] + R[i1 + 2 * i2])) +
              18.0 * exp(6.0 * eta) * exp(-2.0 * alpha) * S[i2 + 2 * i1];
        }
      }
    }
  }

  delete[] dP0;

  return;
}

void fast_spectra::compute_delta_bispectra(void) {
  FILE *fp;
  char base[256], fname[256];
  int offset, ind, ik_s, nk_s, nr_s, nx_s, ik_e, nk_e, nr_e, nx_e;
  double ki, qi, ri, xi, kpi, p_i, k1, k2, k3, alpha_k1, alpha_k2, alpha_k3;
  double Pk1, Pk2, Pk3, dPk1, dPk2, dPk3;
  double G1reg[6], G2reg[6], dG1_1l[6], dG2_1l[6], dG1reg[6], dG2reg[6];
  double T[8], U[8], V[8 * 3], dBcorr211_I[8], dBcorr211_II[8], dBcorr222[8],
      dBcorr321[8];
  double P0[3], G1_1l[6], G2_1l[6], F2[8], Bcorr222[8], Bcorr321[8];
  double *dP0;

  sprintf(base, "%s/diagram_bispec/diagram_bispec", kernel_root.c_str());

  offset = 2 * nq;

  dP0 = new double[nq];

  for (int iq = 0; iq < nq; ++iq) {
    qi = q[iq];
    dP0[iq] = spec->P0(qi) - (c * fidP0(qi));
  }

  for (int ik = 0; ik < nk; ++ik) {
    sprintf(fname, "%s_k%03d.dat", base, ik);

    if ((fp = fopen(fname, "rb")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    fread(&ik_s, sizeof(int), 1, fp);
    fread(&nk_s, sizeof(int), 1, fp);
    fread(&nr_s, sizeof(int), 1, fp);
    fread(&nx_s, sizeof(int), 1, fp);
    fread(&ki, sizeof(double), 1, fp);

    if (ik_s != ik || nk_s != nk || nr_s != nr || nx_s != nx) {
      printf("diagram data load failed!:%d\n", ik);
      exit(1);
    }

    load_kernels_bispec(ik);

    for (int ir = 0; ir < nr; ++ir) {
      for (int ix = 0; ix < nx; ++ix) {
        fread(&ri, sizeof(double), 1, fp);
        fread(&xi, sizeof(double), 1, fp);
        fread(P0, sizeof(double), 3, fp);
        fread(G1_1l, sizeof(double), 6, fp);
        fread(G2_1l, sizeof(double), 6, fp);
        fread(F2, sizeof(double), 6, fp);
        fread(Bcorr222, sizeof(double), 8, fp);
        fread(Bcorr321, sizeof(double), 8, fp);

        kpi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p_i = ki * ri;

        k1 = p_i;
        k2 = kpi;
        k3 = ki;

        Pk1 = P0[0];
        Pk2 = P0[1];
        Pk3 = P0[2];

        dPk1 = spec->P0(k1) - (c * Pk1);
        dPk2 = spec->P0(k2) - (c * Pk2);
        dPk3 = spec->P0(k3) - (c * Pk3);

        alpha_k1 = 0.5 * exp(2.0 * eta) * sqr(k1) * sigmad2_b(k1);
        alpha_k2 = 0.5 * exp(2.0 * eta) * sqr(k2) * sigmad2_b(k2);
        alpha_k3 = 0.5 * exp(2.0 * eta) * sqr(k3) * sigmad2_b(k3);

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
                  L1[offset + j + 3 * (i + 2 * (iq + nq * (ix + nx * ir)))] *
                  dP0[iq];
              dG2_1l[j + 3 * i] +=
                  wq[iq] * cub(qi) *
                  N2[j + 3 * (i + 2 * (iq + nq * (ix + nx * ir)))] * dP0[iq];
            }
          }

          for (int i = 0; i < 8; ++i) {
            T[i] += wq[iq] * cub(qi) *
                    (sqr(c) * T3[i + 8 * (iq + nq * (ix + nx * ir))]) * dP0[iq];
            U[i] += wq[iq] * cub(qi) *
                    (sqr(c) * U3[i + 8 * (iq + nq * (ix + nx * ir))]) * dP0[iq];
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
            V[j + 3 * i] = sqr(c) * V3[j + 3 * (i + 8 * (ix + nx * ir))];
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

          dBk_1l[i + 8 * (ix + nx * (ir + nr * ik))] =
              dBcorr211_I[i] + dBcorr211_II[i] + dBcorr222[i] + dBcorr321[i];
        }
      }
    }

    fread(&ik_e, sizeof(int), 1, fp);
    fread(&nk_e, sizeof(int), 1, fp);
    fread(&nr_e, sizeof(int), 1, fp);
    fread(&nx_e, sizeof(int), 1, fp);

    if (ik_s != ik_e || nk_s != nk_e || nr_s != nr_e || nx_s != nx_e) {
      printf("diagram data load failed!:%d\n", ik);
      exit(1);
    }

    fclose(fp);
  }

  delete[] dP0;

  return;
}

void fast_spectra::Aterm_recon(void) {
  double rmin, rmax, xmin, xmax;
  double ki, ri, xi, wr, wx;
  double B1_tdd, B1_tdt, B1_ttd, B1_ttt, B2_tdd, B2_tdt, B2_ttd, B2_ttt;

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];
    for (int n = 1; n <= 3; n++)
      A[n - 1 + 3 * ik] = 0.0;
    for (int ir = 0; ir < nr; ir++) {
      rmin = qmin / ki;
      rmax = qmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ix++) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);

        /*
        ddd: 0
        ddt: 1
        dtd: 2
        dtt: 3
        tdd: 4
        tdt: 5
        ttd: 6
        ttt: 7
        */

        B1_tdd = Bkfid_1l[4 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[4 + 8 * (ix + nx * (ir + nr * ik))];
        B1_tdt = Bkfid_1l[5 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[5 + 8 * (ix + nx * (ir + nr * ik))];
        B1_ttd = Bkfid_1l[6 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[6 + 8 * (ix + nx * (ir + nr * ik))];
        B1_ttt = Bkfid_1l[7 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[7 + 8 * (ix + nx * (ir + nr * ik))];
        B2_tdd = Bkfid_1l[2 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[2 + 8 * (ix + nx * (ir + nr * ik))];
        B2_tdt = Bkfid_1l[3 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[3 + 8 * (ix + nx * (ir + nr * ik))];
        B2_ttd = Bkfid_1l[6 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[6 + 8 * (ix + nx * (ir + nr * ik))];
        B2_ttt = Bkfid_1l[7 + 8 * (ix + nx * (ir + nr * ik))] +
                 dBk_1l[7 + 8 * (ix + nx * (ir + nr * ik))];

        for (int n = 1; n <= 3; n++) {
          A[n - 1 + 3 * ik] += wr * wx *
                               (beta * A_func(n, 1, 1, ri, xi) * B1_tdd +
                                beta * At_func(n, 1, 1, ri, xi) * B2_tdd +
                                sqr(beta) * A_func(n, 1, 2, ri, xi) * B1_tdt +
                                sqr(beta) * At_func(n, 1, 2, ri, xi) * B2_tdt +
                                sqr(beta) * A_func(n, 2, 1, ri, xi) * B1_ttd +
                                sqr(beta) * At_func(n, 2, 1, ri, xi) * B2_ttd +
                                cub(beta) * A_func(n, 2, 2, ri, xi) * B1_ttt +
                                cub(beta) * At_func(n, 2, 2, ri, xi) * B2_ttt) *
                               ri * 2.0;
        }
      }
    }
    for (int n = 1; n <= 3; n++)
      A[n - 1 + 3 * ik] *= cub(ki) / sqr(2.0 * pi);
  }

  return;
}

void fast_spectra::Bterm_recon(void) {
  double rmin, rmax, xmin, xmax;
  double ki, ri, xi, wr, wx, k1, k2, k3;
  double logk1, logk2;
  double P1_dt, P1_tt, P2_dt, P2_tt;
  double *logk_table, *dt_table, *tt_table;
  gsl_spline *spl_Pk_1l_dt, *spl_Pk_1l_tt;
  gsl_interp_accel *acc_dt, *acc_tt;

  if (direct_Bterm) {
    logk_table = new double[nk_spl];
    dt_table = new double[nk_spl];
    tt_table = new double[nk_spl];

    acc_dt = gsl_interp_accel_alloc();
    acc_tt = gsl_interp_accel_alloc();

    for (int i = 0; i < nk_spl; i++) {
      logk_table[i] = (log(kmax) - log(kmin)) / (nk_spl - 1.0) * i + log(kmin);
      dt_table[i] = spec->Preg_1loop(DENS, VELO, exp(logk_table[i]));
      tt_table[i] = spec->Preg_1loop(VELO, VELO, exp(logk_table[i]));
    }

    spl_Pk_1l_dt = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
    spl_Pk_1l_tt = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
    gsl_spline_init(spl_Pk_1l_dt, logk_table, dt_table, nk_spl);
    gsl_spline_init(spl_Pk_1l_tt, logk_table, tt_table, nk_spl);

    delete[] logk_table;
    delete[] dt_table;
    delete[] tt_table;
  }

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];
    for (int n = 1; n <= 4; n++)
      B[n - 1 + 4 * ik] = 0.0;

    for (int ir = 0; ir < nr; ++ir) {
      rmin = kmin / ki;
      rmax = kmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);
        k1 = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        k2 = ki * ri;
        k3 = 1.0 + sqr(ri) - 2.0 * ri * xi;

        if (direct_Bterm) {
          logk1 = log(k1);
          logk2 = log(k2);

          if (logk1 > log(kmin) && logk1 < log(kmax)) {
            P1_dt = gsl_spline_eval(spl_Pk_1l_dt, logk1, acc_dt);
            P1_tt = gsl_spline_eval(spl_Pk_1l_tt, logk1, acc_tt);
          }
          else{
            P1_dt = 0.0;
            P1_tt = 0.0;
          }

          if (logk2 > log(kmin) && logk2 < log(kmax)){
            P2_dt = gsl_spline_eval(spl_Pk_1l_dt, logk2, acc_dt);
            P2_tt = gsl_spline_eval(spl_Pk_1l_tt, logk2, acc_tt);
          }
          else {
            P2_dt = 0.0;
            P2_tt = 0.0;
          }
        } else {
          if (k1 > kmin_spl && k1 < kmax_spl) {
            P1_dt = gsl_spline_eval(spl_Pk_1l[1], k1, acc_Pk_1l[1]);
            P1_tt = gsl_spline_eval(spl_Pk_1l[2], k1, acc_Pk_1l[2]);
          } else {
            P1_dt = 0.0;
            P1_tt = 0.0;
          }

          if (k2 > kmin_spl && k2 < kmax_spl) {
            P2_dt = gsl_spline_eval(spl_Pk_1l[1], k2, acc_Pk_1l[1]);
            P2_tt = gsl_spline_eval(spl_Pk_1l[2], k2, acc_Pk_1l[2]);
          } else {
            P2_dt = 0.0;
            P2_tt = 0.0;
          }
        }

        for (int n = 1; n <= 4; n++) {
          B[n - 1 + 4 * ik] +=
              wr * wx *
              (sqr(-beta) * B_func(n, 1, 1, ri, xi) * P1_dt * P2_dt / k3 +
               cub(-beta) * B_func(n, 1, 2, ri, xi) * P1_dt * P2_tt / k3 +
               cub(-beta) * B_func(n, 2, 1, ri, xi) * P1_tt * P2_dt / sqr(k3) +
               qua(-beta) * B_func(n, 2, 2, ri, xi) * P1_tt * P2_tt / sqr(k3)) *
              ri * 2.0;
        }
      }
    }
    for (int n = 1; n <= 4; n++)
      B[n - 1 + 4 * ik] *= cub(ki) / sqr(2.0 * pi);
  }

  if (direct_Bterm) {
    gsl_interp_accel_free(acc_dt);
    gsl_interp_accel_free(acc_tt);
    gsl_spline_free(spl_Pk_1l_dt);
    gsl_spline_free(spl_Pk_1l_tt);
  }

  return;
}

void fast_spectra::construct_spline_spectra(void) {
  double *recon_Pk_1l, *recon_Pk_2l;

  for (int i = 0; i < 3; ++i) {
    acc_Pk_1l[i] = gsl_interp_accel_alloc();
    acc_Pk_2l[i] = gsl_interp_accel_alloc();
    spl_Pk_1l[i] = gsl_spline_alloc(gsl_interp_cspline, nk);
    spl_Pk_2l[i] = gsl_spline_alloc(gsl_interp_cspline, nk);
  }

  recon_Pk_1l = new double[nk];
  recon_Pk_2l = new double[nk];

  for (int i = 0; i < 3; ++i) {
    for (int ik = 0; ik < nk; ++ik) {
      recon_Pk_1l[ik] = Pkfid_1l[i + 3 * ik] + dPk_1l[i + 3 * ik];
      recon_Pk_2l[ik] = Pkfid_2l[i + 3 * ik] + dPk_2l[i + 3 * ik];
    }
    gsl_spline_init(spl_Pk_1l[i], k, recon_Pk_1l, nk);
    gsl_spline_init(spl_Pk_2l[i], k, recon_Pk_2l, nk);
  }

  delete[] recon_Pk_1l;
  delete[] recon_Pk_2l;

  return;
}

void fast_spectra::construct_spline_Aterm(void) {
  double *recon_A;

  for (int i = 0; i < 3; ++i) {
    acc_A[i] = gsl_interp_accel_alloc();
    spl_A[i] = gsl_spline_alloc(gsl_interp_cspline, nk);
  }

  recon_A = new double[nk];

  for (int i = 0; i < 3; ++i) {
    for (int ik = 0; ik < nk; ++ik) {
      recon_A[ik] = A[i + 3 * ik];
    }
    gsl_spline_init(spl_A[i], k, recon_A, nk);
  }

  delete[] recon_A;

  return;
}

void fast_spectra::construct_spline_Bterm(void) {
  double *recon_B;

  for (int i = 0; i < 4; ++i) {
    acc_B[i] = gsl_interp_accel_alloc();
    spl_B[i] = gsl_spline_alloc(gsl_interp_cspline, nk);
  }

  recon_B = new double[nk];

  for (int i = 0; i < 4; ++i) {
    for (int ik = 0; ik < nk; ++ik) {
      recon_B[ik] = B[i + 4 * ik];
    }
    gsl_spline_init(spl_B[i], k, recon_B, nk);
  }

  delete[] recon_B;

  return;
}

map<string, double> fast_spectra::get_spectra_1l(double k0) {
  int n;
  map<string, double> res;

  if (k0 < kmin_spl) {
    res["dd"] = cosmo->Plin(k0);
    res["dt"] = cosmo->Plin(k0);
    res["tt"] = cosmo->Plin(k0);
  } else if (k0 > kmax_spl) {
    res["dd"] = 0.0;
    res["dt"] = 0.0;
    res["tt"] = 0.0;
  } else {
    res["dd"] = gsl_spline_eval(spl_Pk_1l[0], k0, acc_Pk_1l[0]);
    res["dt"] = gsl_spline_eval(spl_Pk_1l[1], k0, acc_Pk_1l[1]);
    res["tt"] = gsl_spline_eval(spl_Pk_1l[2], k0, acc_Pk_1l[2]);
  }

  return res;
}

map<string, double> fast_spectra::get_spectra_2l(double k0) {
  int n;
  map<string, double> res;

  if (k0 < kmin_spl) {
    res["dd"] = cosmo->Plin(k0);
    res["dt"] = cosmo->Plin(k0);
    res["tt"] = cosmo->Plin(k0);
  } else if (k0 > kmax_spl) {
    res["dd"] = 0.0;
    res["dt"] = 0.0;
    res["tt"] = 0.0;
  } else {
    res["dd"] = gsl_spline_eval(spl_Pk_2l[0], k0, acc_Pk_2l[0]);
    res["dt"] = gsl_spline_eval(spl_Pk_2l[1], k0, acc_Pk_2l[1]);
    res["tt"] = gsl_spline_eval(spl_Pk_2l[2], k0, acc_Pk_2l[2]);
  }

  return res;
}

map<string, double> fast_spectra::get_Aterm(double k0) {
  int n;
  vector<double> A2, A4, A6;
  map<string, double> res;

  if (k0 < kmin_spl || k0 > kmax_spl) {
    res["A2"] = 0.0;
    res["A4"] = 0.0;
    res["A6"] = 0.0;
  } else {
    res["A2"] = gsl_spline_eval(spl_A[0], k0, acc_A[0]);
    res["A4"] = gsl_spline_eval(spl_A[1], k0, acc_A[1]);
    res["A6"] = gsl_spline_eval(spl_A[2], k0, acc_A[2]);
  }

  return res;
}

map<string, double> fast_spectra::get_Bterm(double k0) {
  int n;
  map<string, double> res;

  if (k0 < kmin_spl || k0 > kmax_spl) {
    res["B2"] = 0.0;
    res["B4"] = 0.0;
    res["B6"] = 0.0;
    res["B8"] = 0.0;
  } else {
    res["B2"] = gsl_spline_eval(spl_B[0], k0, acc_B[0]);
    res["B4"] = gsl_spline_eval(spl_B[1], k0, acc_B[1]);
    res["B6"] = gsl_spline_eval(spl_B[2], k0, acc_B[2]);
    res["B8"] = gsl_spline_eval(spl_B[3], k0, acc_B[3]);
  }

  return res;
}

map<string, vector<double>> fast_spectra::get_spectra_1l(vector<double> k0) {
  int n;
  vector<double> Pdd, Pdt, Ptt;
  map<string, vector<double>> res;

  n = k0.size();

  Pdd.resize(n);
  Pdt.resize(n);
  Ptt.resize(n);

  for (int i = 0; i < n; ++i) {
    if (k0[i] < kmin_spl) {
      Pdd[i] = cosmo->Plin(k0[i]);
      Pdt[i] = cosmo->Plin(k0[i]);
      Ptt[i] = cosmo->Plin(k0[i]);
    } else if (k0[i] > kmax_spl) {
      Pdd[i] = 0.0;
      Pdt[i] = 0.0;
      Ptt[i] = 0.0;
    } else {
      Pdd[i] = gsl_spline_eval(spl_Pk_1l[0], k0[i], acc_Pk_1l[0]);
      Pdt[i] = gsl_spline_eval(spl_Pk_1l[1], k0[i], acc_Pk_1l[1]);
      Ptt[i] = gsl_spline_eval(spl_Pk_1l[2], k0[i], acc_Pk_1l[2]);
    }
  }

  res["dd"] = Pdd;
  res["dt"] = Pdt;
  res["tt"] = Ptt;

  return res;
}

map<string, vector<double>> fast_spectra::get_spectra_2l(vector<double> k0) {
  int n;
  vector<double> Pdd, Pdt, Ptt;
  map<string, vector<double>> res;

  n = k0.size();

  Pdd.resize(n);
  Pdt.resize(n);
  Ptt.resize(n);

  for (int i = 0; i < n; ++i) {
    if (k0[i] < kmin_spl) {
      Pdd[i] = cosmo->Plin(k0[i]);
      Pdt[i] = cosmo->Plin(k0[i]);
      Ptt[i] = cosmo->Plin(k0[i]);
    } else if (k0[i] > kmax_spl) {
      Pdd[i] = 0.0;
      Pdt[i] = 0.0;
      Ptt[i] = 0.0;
    } else {
      Pdd[i] = gsl_spline_eval(spl_Pk_2l[0], k0[i], acc_Pk_2l[0]);
      Pdt[i] = gsl_spline_eval(spl_Pk_2l[1], k0[i], acc_Pk_2l[1]);
      Ptt[i] = gsl_spline_eval(spl_Pk_2l[2], k0[i], acc_Pk_2l[2]);
    }
  }

  res["dd"] = Pdd;
  res["dt"] = Pdt;
  res["tt"] = Ptt;

  return res;
}

map<string, vector<double>> fast_spectra::get_Aterm(vector<double> k0) {
  int n;
  vector<double> A2, A4, A6;
  map<string, vector<double>> res;

  n = k0.size();

  A2.resize(n);
  A4.resize(n);
  A6.resize(n);

  for (int i = 0; i < n; ++i) {
    if (k0[i] < kmin_spl || k0[i] > kmax_spl) {
      A2[i] = 0.0;
      A4[i] = 0.0;
      A6[i] = 0.0;
    } else {
      A2[i] = gsl_spline_eval(spl_A[0], k0[i], acc_A[0]);
      A4[i] = gsl_spline_eval(spl_A[1], k0[i], acc_A[1]);
      A6[i] = gsl_spline_eval(spl_A[2], k0[i], acc_A[2]);
    }
  }

  res["A2"] = A2;
  res["A4"] = A4;
  res["A6"] = A6;

  return res;
}

map<string, vector<double>> fast_spectra::get_Bterm(vector<double> k0) {
  int n;
  vector<double> B2, B4, B6, B8;
  map<string, vector<double>> res;

  n = k0.size();

  B2.resize(n);
  B4.resize(n);
  B6.resize(n);
  B8.resize(n);

  for (int i = 0; i < n; ++i) {
    if (k0[i] < kmin_spl || k0[i] > kmax_spl) {
      B2[i] = 0.0;
      B4[i] = 0.0;
      B6[i] = 0.0;
      B8[i] = 0.0;
    } else {
      B2[i] = gsl_spline_eval(spl_B[0], k0[i], acc_B[0]);
      B4[i] = gsl_spline_eval(spl_B[1], k0[i], acc_B[1]);
      B6[i] = gsl_spline_eval(spl_B[2], k0[i], acc_B[2]);
      B8[i] = gsl_spline_eval(spl_B[3], k0[i], acc_B[3]);
    }
  }

  res["B2"] = B2;
  res["B4"] = B4;
  res["B6"] = B6;
  res["B8"] = B8;

  return res;
}

/* auxiliary functions for A term */
double fast_spectra::A_func(int a, int b, int c, double r, double x) {
  if ((a == 1 && b == 1 && c == 1) || (a == 2 && b == 1 && c == 2)) {
    return r * x;
  } else if ((a == 1 && b == 2 && c == 1) || (a == 2 && b == 2 && c == 2)) {
    return -r * r * (-2.0 + 3.0 * r * x) * (x * x - 1.0) /
           (2.0 * (1.0 + r * r - 2.0 * r * x));
  } else if ((a == 2 && b == 2 && c == 1) || (a == 3 && b == 2 && c == 2)) {
    return r *
           (2.0 * x + r * (2.0 - 6.0 * x * x) +
            r * r * x * (-3.0 + 5.0 * x * x)) /
           (2.0 * (1.0 + r * r - 2.0 * r * x));
  } else
    return 0.0;
}

double fast_spectra::At_func(int a, int b, int c, double r, double x) {
  if ((a == 1 && b == 1 && c == 1) || (a == 2 && b == 1 && c == 2)) {
    return -r * r * (r * x - 1.0) / (1.0 + r * r - 2.0 * r * x);
  } else if ((a == 1 && b == 2 && c == 1) || (a == 2 && b == 2 && c == 2)) {
    return r * r * (-1.0 + 3.0 * r * x) * (x * x - 1.0) /
           (2.0 * (1.0 + r * r - 2.0 * r * x));
  } else if ((a == 2 && b == 2 && c == 1) || (a == 3 && b == 2 && c == 2)) {
    return -r * r * (1.0 - 3.0 * x * x + r * x * (-3.0 + 5.0 * x * x)) /
           (2.0 * (1.0 + r * r - 2.0 * r * x));
  } else
    return 0.0;
}

/* auxiliary functions for B term */
double fast_spectra::B_func(int a, int b, int c, double r, double x) {
  if (a == 1 && b == 1 && c == 1) {
    return r * r / 2.0 * (x * x - 1.0);
  } else if (a == 1 && b == 1 && c == 2) {
    return 3.0 * r * r / 8.0 * sqr(x * x - 1.0);
  } else if (a == 1 && b == 2 && c == 1) {
    return 3.0 * qua(r) / 8.0 * sqr(x * x - 1.0);
  } else if (a == 1 && b == 2 && c == 2) {
    return 5.0 * qua(r) / 16.0 * cub(x * x - 1.0);
  } else if (a == 2 && b == 1 && c == 1) {
    return r / 2.0 * (r + 2.0 * x - 3.0 * r * x * x);
  } else if (a == 2 && b == 1 && c == 2) {
    return -3.0 * r / 4.0 * (x * x - 1.0) * (-r - 2.0 * x + 5.0 * r * x * x);
  } else if (a == 2 && b == 2 && c == 1) {
    return 3.0 * r * r / 4.0 * (x * x - 1.0) *
           (-2.0 + r * r + 6.0 * r * x - 5.0 * r * r * x * x);
  } else if (a == 2 && b == 2 && c == 2) {
    return -3.0 * r * r / 16.0 * sqr(x * x - 1.0) *
           (6.0 - 30.0 * r * x - 5.0 * r * r + 35.0 * r * r * x * x);
  } else if (a == 3 && b == 1 && c == 2) {
    return r / 8.0 *
           (4.0 * x * (3.0 - 5.0 * x * x) +
            r * (3.0 - 30.0 * x * x + 35.0 * qua(x)));
  } else if (a == 3 && b == 2 && c == 1) {
    return r / 8.0 *
           (-8.0 * x +
            r * (-12.0 + 36.0 * x * x + 12.0 * r * x * (3.0 - 5.0 * x * x) +
                 r * r * (3.0 - 30.0 * x * x + 35.0 * qua(x))));
  } else if (a == 3 && b == 2 && c == 2) {
    return 3.0 * r / 16.0 * (x * x - 1.0) *
           (-8.0 * x +
            r * (-12.0 + 60.0 * x * x + 20.0 * r * x * (3.0 - 7.0 * x * x) +
                 5.0 * r * r * (1.0 - 14.0 * x * x + 21.0 * qua(x))));
  } else if (a == 4 && b == 2 && c == 2) {
    return r / 16.0 *
           (8.0 * x * (-3.0 + 5.0 * x * x) -
            6.0 * r * (3.0 - 30.0 * x * x + 35.0 * qua(x)) +
            6.0 * r * r * x * (15.0 - 70.0 * x * x + 63.0 * qua(x)) +
            cub(r) *
                (5.0 - 21.0 * x * x * (5.0 - 15.0 * x * x + 11.0 * qua(x))));
  } else
    return 0.0;
}
