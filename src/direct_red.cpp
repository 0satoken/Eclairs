#include "direct_red.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <map>

direct_red::direct_red(params &params, cosmology &cosmo, spectra &spec) {
  this->spec = &spec;

  f = cosmo.get_growth_rate();
  eta = cosmo.get_eta();
  pi = 4.0 * atan(1.0);

  lambda_p = params.dparams["lambda_power"];
  lambda_b = params.dparams["lambda_bispectrum"];

  /* bias parameter */
  b1 = params.dparams["b1"];
  beta = f / b1;

  /* flags */
  flag_SPT = params.bparams["direct_SPT"];
  flag_1loop = params.bparams["direct_1loop"];

  /* integration setting */
  kmin = params.dparams["kmin"];
  kmax = params.dparams["kmax"];

  nr = params.iparams["direct_nr"];
  nx = params.iparams["direct_nx"];
  nq = params.iparams["direct_nq"];
  nmu = params.iparams["direct_nmu"];
  nphi = params.iparams["direct_nphi"];
  mumin = params.dparams["direct_mumin"];
  mumax = params.dparams["direct_mumax"];
  phimin = params.dparams["direct_phimin"];
  phimax = params.dparams["direct_phimax"];

  t_r = gsl_integration_glfixed_table_alloc(nr);
  t_x = gsl_integration_glfixed_table_alloc(nx);
  t_q = gsl_integration_glfixed_table_alloc(nq);
  t_mu = gsl_integration_glfixed_table_alloc(nmu);
  t_phi = gsl_integration_glfixed_table_alloc(nphi);

  q = new double[nq];
  mu = new double[nmu];
  phi = new double[nphi];
  wq = new double[nq];
  wmu = new double[nmu];
  wphi = new double[nphi];

  for (int iq = 0; iq < nq; iq++) {
    gsl_integration_glfixed_point(log(kmin), log(kmax), iq, &q[iq], &wq[iq], t_q);
    q[iq] = exp(q[iq]);
  }

  for (int imu = 0; imu < nmu; imu++) {
    gsl_integration_glfixed_point(mumin, mumax, imu, &mu[imu], &wmu[imu], t_mu);
  }

  for (int iphi = 0; iphi < nphi; iphi++) {
    gsl_integration_glfixed_point(phimin, phimax, iphi, &phi[iphi], &wphi[iphi], t_phi);
  }

  /* Monte-Carlo integration */
  flag_MC = params.bparams["direct_MC"];
  if (flag_MC) {
    dim = 5;
    MC_calls = params.iparams["direct_MC_calls"];
    MC_tol = params.dparams["direct_MC_tol"];
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    s = gsl_monte_vegas_alloc(dim);
  }

  /* setting spline functions of sigmad and power spectra at 1-loop for B-term
   */
  nk_spl = params.iparams["nk_spl"];
  set_Pk1l_spline();
  set_sigmad2_spline();

  /* spline */
  flag_spline = params.bparams["direct_spline"];

  if (flag_spline) {
    acc_Pk = new gsl_interp_accel *[3];
    acc_A = new gsl_interp_accel *[3];
    acc_B = new gsl_interp_accel *[4];

    spl_Pk = new gsl_spline *[3];
    spl_A = new gsl_spline *[3];
    spl_B = new gsl_spline *[4];

    if (flag_SPT) {
      acc_C = new gsl_interp_accel *[4];
      spl_C = new gsl_spline *[4];
    }
    set_all_spline();
  }
}

direct_red::~direct_red() {
  gsl_integration_glfixed_table_free(t_r);
  gsl_integration_glfixed_table_free(t_x);
  gsl_integration_glfixed_table_free(t_q);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_integration_glfixed_table_free(t_phi);
  delete[] q;
  delete[] mu;
  delete[] phi;
  delete[] wq;
  delete[] wmu;
  delete[] wphi;
  gsl_interp_accel_free(acc_Pk1l_dd);
  gsl_interp_accel_free(acc_Pk1l_dt);
  gsl_interp_accel_free(acc_Pk1l_tt);
  gsl_interp_accel_free(acc_sigmad2_p);
  gsl_interp_accel_free(acc_sigmad2_b);
  gsl_spline_free(spl_Pk1l_dd);
  gsl_spline_free(spl_Pk1l_dt);
  gsl_spline_free(spl_Pk1l_tt);
  gsl_spline_free(spl_sigmad2_p);
  gsl_spline_free(spl_sigmad2_b);

  if (flag_MC) {
    gsl_monte_vegas_free(s);
    gsl_rng_free(r);
  }

  if (flag_spline) {
    for (int i = 0; i < 3; ++i) {
      gsl_spline_free(spl_Pk[i]);
      gsl_interp_accel_free(acc_Pk[i]);
    }

    for (int i = 0; i < 3; ++i) {
      gsl_spline_free(spl_A[i]);
      gsl_interp_accel_free(acc_A[i]);
    }

    for (int i = 0; i < 4; ++i) {
      gsl_spline_free(spl_B[i]);
      gsl_interp_accel_free(acc_B[i]);
    }

    delete[] spl_Pk;
    delete[] spl_A;
    delete[] spl_B;
    delete[] acc_Pk;
    delete[] acc_A;
    delete[] acc_B;

    if (flag_SPT) {
      for (int i = 0; i < 4; ++i) {
        gsl_spline_free(spl_C[i]);
        gsl_interp_accel_free(acc_C[i]);
      }
      delete[] spl_C;
      delete[] acc_C;
    }
  }
}

void direct_red::set_all_spline(void) {
  double *k_, *dd_, *dt_, *tt_;
  vector<double> kvec;
  map<string, vector<double>> res_A, res_B, res_C;

  for (int i = 0; i < 3; ++i) {
    acc_Pk[i] = gsl_interp_accel_alloc();
    spl_Pk[i] = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  }

  k_ = new double[nk_spl];
  dd_ = new double[nk_spl];
  dt_ = new double[nk_spl];
  tt_ = new double[nk_spl];
  kvec.resize(nk_spl);

  for (int i = 0; i < nk_spl; i++) {
    k_[i] = (log(kmax) - log(kmin)) / (nk_spl - 1) * i + log(kmin);
    k_[i] = exp(k_[i]);
    kvec[i] = k_[i];

    if (flag_1loop) {
      if (flag_SPT) {
        dd_[i] = spec->Pspt_1loop(DENS, DENS, k_[i]);
        dt_[i] = spec->Pspt_1loop(DENS, VELO, k_[i]);
        tt_[i] = spec->Pspt_1loop(VELO, VELO, k_[i]);
      } else {
        dd_[i] = spec->Preg_1loop(DENS, DENS, k_[i]);
        dt_[i] = spec->Preg_1loop(DENS, VELO, k_[i]);
        tt_[i] = spec->Preg_1loop(VELO, VELO, k_[i]);
      }
    } else {
      if (flag_SPT) {
        dd_[i] = spec->Pspt_2loop(DENS, DENS, k_[i]);
        dt_[i] = spec->Pspt_2loop(DENS, VELO, k_[i]);
        tt_[i] = spec->Pspt_2loop(VELO, VELO, k_[i]);
      } else {
        dd_[i] = spec->Preg_2loop(DENS, DENS, k_[i]);
        dt_[i] = spec->Preg_2loop(DENS, VELO, k_[i]);
        tt_[i] = spec->Preg_2loop(VELO, VELO, k_[i]);
      }
    }
  }

  gsl_spline_init(spl_Pk[0], k_, dd_, nk_spl);
  gsl_spline_init(spl_Pk[1], k_, dt_, nk_spl);
  gsl_spline_init(spl_Pk[2], k_, tt_, nk_spl);

  res_A = get_Aterm(kvec);

  for (int i = 0; i < 3; ++i) {
    acc_A[i] = gsl_interp_accel_alloc();
    spl_A[i] = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  }

  gsl_spline_init(spl_A[0], k_, &res_A["A2"][0], nk_spl);
  gsl_spline_init(spl_A[1], k_, &res_A["A4"][0], nk_spl);
  gsl_spline_init(spl_A[2], k_, &res_A["A6"][0], nk_spl);

  res_B = get_Bterm(kvec);

  for (int i = 0; i < 4; ++i) {
    acc_B[i] = gsl_interp_accel_alloc();
    spl_B[i] = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  }
  gsl_spline_init(spl_B[0], k_, &res_B["B2"][0], nk_spl);
  gsl_spline_init(spl_B[1], k_, &res_B["B4"][0], nk_spl);
  gsl_spline_init(spl_B[2], k_, &res_B["B6"][0], nk_spl);
  gsl_spline_init(spl_B[3], k_, &res_B["B8"][0], nk_spl);

  if (flag_SPT) {
    res_C = get_Cterm(kvec);

    for (int i = 0; i < 4; ++i) {
      acc_C[i] = gsl_interp_accel_alloc();
      spl_C[i] = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
    }

    gsl_spline_init(spl_C[0], k_, &res_C["C2"][0], nk_spl);
    gsl_spline_init(spl_C[1], k_, &res_C["C4"][0], nk_spl);
    gsl_spline_init(spl_C[2], k_, &res_C["C6"][0], nk_spl);
    gsl_spline_init(spl_C[3], k_, &res_C["C8"][0], nk_spl);
  }

  delete[] k_;
  delete[] dd_;
  delete[] dt_;
  delete[] tt_;

  return;
}

void direct_red::set_Pk1l_spline(void) {
  double *k_, *dd_, *dt_, *tt_;

  k_ = new double[nk_spl];
  dd_ = new double[nk_spl];
  dt_ = new double[nk_spl];
  tt_ = new double[nk_spl];

  acc_Pk1l_dd = gsl_interp_accel_alloc();
  acc_Pk1l_dt = gsl_interp_accel_alloc();
  acc_Pk1l_tt = gsl_interp_accel_alloc();

  for (int i = 0; i < nk_spl; i++) {
    k_[i] = (log(kmax) - log(kmin)) / (nk_spl - 1) * i + log(kmin);
    k_[i] = exp(k_[i]);

    if (flag_SPT) {
      dd_[i] = spec->Pspt_1loop(DENS, DENS, k_[i]);
      dt_[i] = spec->Pspt_1loop(DENS, VELO, k_[i]);
      tt_[i] = spec->Pspt_1loop(VELO, VELO, k_[i]);
    } else {
      dd_[i] = spec->Preg_1loop(DENS, DENS, k_[i]);
      dt_[i] = spec->Preg_1loop(DENS, VELO, k_[i]);
      tt_[i] = spec->Preg_1loop(VELO, VELO, k_[i]);
    }
  }

  spl_Pk1l_dd = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  spl_Pk1l_dt = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  spl_Pk1l_tt = gsl_spline_alloc(gsl_interp_cspline, nk_spl);

  gsl_spline_init(spl_Pk1l_dd, k_, dd_, nk_spl);
  gsl_spline_init(spl_Pk1l_dt, k_, dt_, nk_spl);
  gsl_spline_init(spl_Pk1l_tt, k_, tt_, nk_spl);

  delete[] k_;
  delete[] dd_;
  delete[] dt_;
  delete[] tt_;

  return;
}

void direct_red::set_sigmad2_spline(void) {
  double *logk_table, *sigmad2_p_table, *sigmad2_b_table;

  logk_table = new double[nk_spl];
  sigmad2_p_table = new double[nk_spl];
  sigmad2_b_table = new double[nk_spl];

  acc_sigmad2_p = gsl_interp_accel_alloc();
  acc_sigmad2_b = gsl_interp_accel_alloc();

  for (int i = 0; i < nk_spl; i++) {
    logk_table[i] =
        (log(kmax) - log(kmin)) / (nk_spl - 1.0) * i + log(kmin);
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

double direct_red::Pk1l_dd_spl(double k) {
  double res;

  if (k < kmin || k > kmax) {
    res = 0.0;
  } else {
    res = gsl_spline_eval(spl_Pk1l_dd, k, acc_Pk1l_dd);
  }

  return res;
}

double direct_red::Pk1l_dt_spl(double k) {
  double res;

  if (k < kmin || k > kmax) {
    res = 0.0;
  } else {
    res = gsl_spline_eval(spl_Pk1l_dt, k, acc_Pk1l_dt);
  }

  return res;
}

double direct_red::Pk1l_tt_spl(double k) {
  double res;

  if (k < kmin || k > kmax) {
    res = 0.0;
  } else {
    res = gsl_spline_eval(spl_Pk1l_tt, k, acc_Pk1l_tt);
  }

  return res;
}

double direct_red::sigmad2_p(double k) {
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

double direct_red::sigmad2_b(double k) {
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

map<string, vector<double>> direct_red::get_spectra(vector<double> k) {
  int nk;
  double ki;
  map<string, vector<double>> res;
  vector<double> Pdd, Pdt, Ptt;

  nk = k.size();
  Pdd.resize(nk);
  Pdt.resize(nk);
  Ptt.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];

    if (flag_SPT) {
      if (flag_1loop) {
        Pdd[ik] = spec->Pspt_1loop(DENS, DENS, ki);
        Pdt[ik] = spec->Pspt_1loop(DENS, VELO, ki);
        Ptt[ik] = spec->Pspt_1loop(VELO, VELO, ki);
      } else {
        Pdd[ik] = spec->Pspt_2loop(DENS, DENS, ki);
        Pdt[ik] = spec->Pspt_2loop(DENS, VELO, ki);
        Ptt[ik] = spec->Pspt_2loop(VELO, VELO, ki);
      }
    } else {
      if (flag_1loop) {
        Pdd[ik] = spec->Preg_1loop(DENS, DENS, ki);
        Pdt[ik] = spec->Preg_1loop(DENS, VELO, ki);
        Ptt[ik] = spec->Preg_1loop(VELO, VELO, ki);
      } else {
        Pdd[ik] = spec->Preg_2loop(DENS, DENS, ki);
        Pdt[ik] = spec->Preg_2loop(DENS, VELO, ki);
        Ptt[ik] = spec->Preg_2loop(VELO, VELO, ki);
      }
    }
  }

  res["dd"] = Pdd;
  res["dt"] = Pdt;
  res["tt"] = Ptt;

  return res;
}

map<string, vector<double>> direct_red::get_Aterm(vector<double> k) {
  map<string, vector<double>> res;

  if (flag_MC) {
    res = Aterm_MC(k);
  } else {
    res = Aterm_direct(k);
  }

  return res;
}

map<string, vector<double>> direct_red::get_Bterm(vector<double> k) {
  map<string, vector<double>> res;

  res = Bterm(k);

  return res;
}

map<string, vector<double>> direct_red::get_Cterm(vector<double> k) {
  map<string, vector<double>> res;

  res = Cterm(k);

  return res;
}

map<string, double> direct_red::get_spectra(double k) {
  map<string, double> res;

  if (flag_SPT) {
    if (flag_1loop) {
      res["dd"] = spec->Pspt_1loop(DENS, DENS, k);
      res["dt"] = spec->Pspt_1loop(DENS, VELO, k);
      res["tt"] = spec->Pspt_1loop(VELO, VELO, k);
    } else {
      res["dd"] = spec->Pspt_2loop(DENS, DENS, k);
      res["dt"] = spec->Pspt_2loop(DENS, VELO, k);
      res["tt"] = spec->Pspt_2loop(VELO, VELO, k);
    }
  } else {
    if (flag_1loop) {
      res["dd"] = spec->Preg_1loop(DENS, DENS, k);
      res["dt"] = spec->Preg_1loop(DENS, VELO, k);
      res["tt"] = spec->Preg_1loop(VELO, VELO, k);
    } else {
      res["dd"] = spec->Preg_2loop(DENS, DENS, k);
      res["dt"] = spec->Preg_2loop(DENS, VELO, k);
      res["tt"] = spec->Preg_2loop(VELO, VELO, k);
    }
  }

  return res;
}

map<string, double> direct_red::get_Aterm(double k) {
  map<string, vector<double>> Aterm_res;
  map<string, double> res;
  vector<double> kvec;

  kvec.push_back(k);

  if (flag_MC) {
    Aterm_res = Aterm_MC(kvec);
  } else {
    Aterm_res = Aterm_direct(kvec);
  }

  res["A2"] = Aterm_res["A2"][0];
  res["A4"] = Aterm_res["A4"][0];
  res["A6"] = Aterm_res["A6"][0];

  return res;
}

map<string, double> direct_red::get_Bterm(double k) {
  map<string, vector<double>> Bterm_res;
  map<string, double> res;
  vector<double> kvec;

  kvec.push_back(k);
  Bterm_res = Bterm(kvec);

  res["B2"] = Bterm_res["B2"][0];
  res["B4"] = Bterm_res["B4"][0];
  res["B6"] = Bterm_res["B6"][0];
  res["B8"] = Bterm_res["B8"][0];

  return res;
}

map<string, double> direct_red::get_Cterm(double k) {
  map<string, vector<double>> Cterm_res;
  map<string, double> res;
  vector<double> kvec;

  kvec.push_back(k);
  Cterm_res = Cterm(kvec);

  res["C2"] = Cterm_res["C2"][0];
  res["C4"] = Cterm_res["C4"][0];
  res["C6"] = Cterm_res["C6"][0];
  res["C8"] = Cterm_res["C8"][0];

  return res;
}

map<string, vector<double>> direct_red::get_spl_spectra(vector<double> k) {
  int nk;
  double ki;
  map<string, vector<double>> res;
  vector<double> Pdd, Pdt, Ptt;

  nk = k.size();
  Pdd.resize(nk);
  Pdt.resize(nk);
  Ptt.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];

    if (ki < kmin || ki > kmax) {
      Pdd[ik] = 0.0;
      Pdt[ik] = 0.0;
      Ptt[ik] = 0.0;
    } else {
      Pdd[ik] = gsl_spline_eval(spl_Pk[0], ki, acc_Pk[0]);
      Pdt[ik] = gsl_spline_eval(spl_Pk[1], ki, acc_Pk[1]);
      Ptt[ik] = gsl_spline_eval(spl_Pk[2], ki, acc_Pk[2]);
    }
  }

  res["dd"] = Pdd;
  res["dt"] = Pdt;
  res["tt"] = Ptt;

  return res;
}

map<string, vector<double>> direct_red::get_spl_Aterm(vector<double> k) {
  int nk;
  double ki;
  map<string, vector<double>> res;
  vector<double> A2, A4, A6;

  nk = k.size();
  A2.resize(nk);
  A4.resize(nk);
  A6.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];

    if (ki < kmin || ki > kmax) {
      A2[ik] = 0.0;
      A4[ik] = 0.0;
      A6[ik] = 0.0;
    } else {
      A2[ik] = gsl_spline_eval(spl_A[0], ki, acc_A[0]);
      A4[ik] = gsl_spline_eval(spl_A[1], ki, acc_A[1]);
      A6[ik] = gsl_spline_eval(spl_A[2], ki, acc_A[2]);
    }
  }

  res["A2"] = A2;
  res["A4"] = A4;
  res["A6"] = A6;

  return res;
}

map<string, vector<double>> direct_red::get_spl_Bterm(vector<double> k) {
  int nk;
  double ki;
  map<string, vector<double>> res;
  vector<double> B2, B4, B6, B8;

  nk = k.size();
  B2.resize(nk);
  B4.resize(nk);
  B6.resize(nk);
  B8.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];

    if (ki < kmin || ki > kmax) {
      B2[ik] = 0.0;
      B4[ik] = 0.0;
      B6[ik] = 0.0;
      B8[ik] = 0.0;
    } else {
      B2[ik] = gsl_spline_eval(spl_B[0], ki, acc_B[0]);
      B4[ik] = gsl_spline_eval(spl_B[1], ki, acc_B[1]);
      B6[ik] = gsl_spline_eval(spl_B[2], ki, acc_B[2]);
      B8[ik] = gsl_spline_eval(spl_B[3], ki, acc_B[3]);
    }
  }

  res["B2"] = B2;
  res["B4"] = B4;
  res["B6"] = B6;
  res["B8"] = B8;

  return res;
}

map<string, vector<double>> direct_red::get_spl_Cterm(vector<double> k) {
  int nk;
  double ki;
  map<string, vector<double>> res;
  vector<double> C2, C4, C6, C8;

  nk = k.size();
  C2.resize(nk);
  C4.resize(nk);
  C6.resize(nk);
  C8.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];

    if (ki < kmin || ki > kmax) {
      C2[ik] = 0.0;
      C4[ik] = 0.0;
      C6[ik] = 0.0;
      C8[ik] = 0.0;
    } else {
      C2[ik] = gsl_spline_eval(spl_C[0], ki, acc_C[0]);
      C4[ik] = gsl_spline_eval(spl_C[1], ki, acc_C[1]);
      C6[ik] = gsl_spline_eval(spl_C[2], ki, acc_C[2]);
      C8[ik] = gsl_spline_eval(spl_C[3], ki, acc_C[3]);
    }
  }

  res["C2"] = C2;
  res["C4"] = C4;
  res["C6"] = C6;
  res["C8"] = C8;

  return res;
}

map<string, double> direct_red::get_spl_spectra(double k) {
  map<string, double> res;

  if (k < kmin || k > kmax) {
    res["dd"] = 0.0;
    res["dt"] = 0.0;
    res["tt"] = 0.0;
  } else {
    res["dd"] = gsl_spline_eval(spl_Pk[0], k, acc_Pk[0]);
    res["dt"] = gsl_spline_eval(spl_Pk[1], k, acc_Pk[1]);
    res["tt"] = gsl_spline_eval(spl_Pk[2], k, acc_Pk[2]);
  }

  return res;
}

map<string, double> direct_red::get_spl_Aterm(double k) {
  map<string, double> res;

  if (k < kmin || k > kmax) {
    res["A2"] = 0.0;
    res["A4"] = 0.0;
    res["A6"] = 0.0;
  } else {
    res["A2"] = gsl_spline_eval(spl_A[0], k, acc_A[0]);
    res["A4"] = gsl_spline_eval(spl_A[1], k, acc_A[1]);
    res["A6"] = gsl_spline_eval(spl_A[2], k, acc_A[2]);
  }

  return res;
}

map<string, double> direct_red::get_spl_Bterm(double k) {
  map<string, double> res;

  if (k < kmin || k > kmax) {
    res["B2"] = 0.0;
    res["B4"] = 0.0;
    res["B6"] = 0.0;
    res["B8"] = 0.0;
  } else {
    res["B2"] = gsl_spline_eval(spl_B[0], k, acc_B[0]);
    res["B4"] = gsl_spline_eval(spl_B[1], k, acc_B[1]);
    res["B6"] = gsl_spline_eval(spl_B[2], k, acc_B[2]);
    res["B8"] = gsl_spline_eval(spl_B[3], k, acc_B[3]);
  }

  return res;
}

map<string, double> direct_red::get_spl_Cterm(double k) {
  map<string, double> res;

  if (k < kmin || k > kmax) {
    res["C2"] = 0.0;
    res["C4"] = 0.0;
    res["C6"] = 0.0;
    res["C8"] = 0.0;
  } else {
    res["C2"] = gsl_spline_eval(spl_C[0], k, acc_C[0]);
    res["C4"] = gsl_spline_eval(spl_C[1], k, acc_C[1]);
    res["C6"] = gsl_spline_eval(spl_C[2], k, acc_C[2]);
    res["C8"] = gsl_spline_eval(spl_C[3], k, acc_C[3]);
  }

  return res;
}

/* A term correction term with direct (no Monte-Carlo) calculation */
map<string, vector<double>> direct_red::Aterm_direct(vector<double> k) {
  int nk;
  double rmin, rmax, xmin, xmax;
  double ki, ri, xi, wri, wxi, p, kp;
  double B1_tdd, B1_tdt, B1_ttd, B1_ttt, B2_tdd, B2_tdt, B2_ttd, B2_ttt;
  double Ai[4];
  map<string, vector<double>> res;
  vector<double> A2, A4, A6;

  nk = k.size();
  A2.resize(nk);
  A4.resize(nk);
  A6.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];
    for (int n = 1; n <= 3; n++)
      Ai[n] = 0.0;

    for (int ir = 0; ir < nr; ++ir) {
      rmin = kmin / ki;
      rmax = kmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wri, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wxi, t_x);
        kp = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p = ki * ri;

        if (flag_1loop) {
          B1_tdd = Bispec_tree(VELO, DENS, DENS, p, kp, ki);
          B1_tdt = Bispec_tree(VELO, DENS, VELO, p, kp, ki);
          B1_ttd = Bispec_tree(VELO, VELO, DENS, p, kp, ki);
          B1_ttt = Bispec_tree(VELO, VELO, VELO, p, kp, ki);
          B2_tdd = Bispec_tree(VELO, DENS, DENS, kp, p, ki);
          B2_tdt = Bispec_tree(VELO, DENS, VELO, kp, p, ki);
          B2_ttd = Bispec_tree(VELO, VELO, DENS, kp, p, ki);
          B2_ttt = Bispec_tree(VELO, VELO, VELO, kp, p, ki);
        } else {
          B1_tdd = Bispec_1loop(VELO, DENS, DENS, p, kp, ki);
          B1_tdt = Bispec_1loop(VELO, DENS, VELO, p, kp, ki);
          B1_ttd = Bispec_1loop(VELO, VELO, DENS, p, kp, ki);
          B1_ttt = Bispec_1loop(VELO, VELO, VELO, p, kp, ki);
          B2_tdd = Bispec_1loop(VELO, DENS, DENS, kp, p, ki);
          B2_tdt = Bispec_1loop(VELO, DENS, VELO, kp, p, ki);
          B2_ttd = Bispec_1loop(VELO, VELO, DENS, kp, p, ki);
          B2_ttt = Bispec_1loop(VELO, VELO, VELO, kp, p, ki);
        }

        for (int n = 1; n <= 3; ++n) {
          Ai[n] += wri * wxi *
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
    A2[ik] = Ai[1] * cub(ki) / sqr(2.0 * pi);
    A4[ik] = Ai[2] * cub(ki) / sqr(2.0 * pi);
    A6[ik] = Ai[3] * cub(ki) / sqr(2.0 * pi);
  }

  res["A2"] = A2;
  res["A4"] = A4;
  res["A6"] = A6;

  return res;
}

/* A term correction with direct (with Monte-Carlo) calculation */
map<string, vector<double>> direct_red::Aterm_MC(vector<double> k) {
  map<string, vector<double>> res, res_Bk211, res_Bk222_Bk321;
  vector<double> A2, A4, A6;
  int nk;

  nk = k.size();
  A2.resize(nk);
  A4.resize(nk);
  A6.resize(nk);

  res_Bk211 = Aterm_Bk211(k);
  res_Bk222_Bk321 = Aterm_Bk222_Bk321(k);

  for (int i = 0; i < nk; ++i) {
    A2[i] = res_Bk211["A2"][i] + res_Bk222_Bk321["A2"][i];
    A4[i] = res_Bk211["A4"][i] + res_Bk222_Bk321["A4"][i];
    A6[i] = res_Bk211["A6"][i] + res_Bk222_Bk321["A6"][i];
  }

  res["A2"] = A2;
  res["A4"] = A4;
  res["A6"] = A6;

  return res;
}

/* A term (Bk211) with direct integration */
map<string, vector<double>> direct_red::Aterm_Bk211(vector<double> k) {
  int nk;
  double rmin, rmax, xmin, xmax;
  double ki, ri, xi, wri, wxi, p, kp;
  double B1_tdd, B1_tdt, B1_ttd, B1_ttt, B2_tdd, B2_tdt, B2_ttd, B2_ttt;
  double Ai[4];
  vector<double> A2, A4, A6;
  map<string, vector<double>> res;

  nk = k.size();
  A2.resize(nk);
  A4.resize(nk);
  A6.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];
    for (int n = 1; n <= 3; ++n)
      Ai[n] = 0.0;

    for (int ir = 0; ir < nr; ++ir) {
      rmin = kmin / ki;
      rmax = kmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wri, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wxi, t_x);
        kp = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        p = ki * ri;
        B1_tdd = Bk211(VELO, DENS, DENS, p, kp, ki);
        B1_tdt = Bk211(VELO, DENS, VELO, p, kp, ki);
        B1_ttd = Bk211(VELO, VELO, DENS, p, kp, ki);
        B1_ttt = Bk211(VELO, VELO, VELO, p, kp, ki);
        B2_tdd = Bk211(VELO, DENS, DENS, kp, p, ki);
        B2_tdt = Bk211(VELO, DENS, VELO, kp, p, ki);
        B2_ttd = Bk211(VELO, VELO, DENS, kp, p, ki);
        B2_ttt = Bk211(VELO, VELO, VELO, kp, p, ki);
        for (int n = 1; n <= 3; ++n) {
          Ai[n] += wri * wxi *
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
    A2[ik] = Ai[1] * cub(ki) / sqr(2.0 * pi);
    A4[ik] = Ai[2] * cub(ki) / sqr(2.0 * pi);
    A6[ik] = Ai[3] * cub(ki) / sqr(2.0 * pi);
  }

  res["A2"] = A2;
  res["A4"] = A4;
  res["A6"] = A6;

  return res;
}

/* A term (Bk222 + Bk321) with Monte-Carlo integration */
map<string, vector<double>> direct_red::Aterm_Bk222_Bk321(vector<double> k) {
  double res_A2, res_A4, res_A6, err;
  double xl[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double xu[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
  int nk;
  Aterm_integral_params par;
  gsl_monte_function G;
  vector<double> A2, A4, A6;
  map<string, vector<double>> res;

  nk = k.size();

  A2.resize(nk);
  A4.resize(nk);
  A6.resize(nk);

  par.eta = eta;
  par.beta = beta;
  par.kmin = kmin;
  par.kmax = kmax;
  par.mumin = mumin;
  par.mumax = mumax;
  par.phimin = phimin;
  par.phimax = phimax;
  par.flag_SPT = flag_SPT;
  par.spec = spec;
  par.spl_sigmad2 = spl_sigmad2_b;
  par.acc_sigmad2 = acc_sigmad2_b;

  G.f = &(direct_red::Aterm_Bk222_Bk321_kernel);
  G.dim = dim;
  G.params = &par;

  for (int ik = 0; ik < nk; ++ik) {
    par.k = k[ik];

    par.n = 1;
    gsl_monte_vegas_init(s);
    gsl_monte_vegas_integrate(&G, xl, xu, dim, 10000, r, s, &res_A2, &err);
    // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
    // s->chisq);

    do {
      gsl_monte_vegas_integrate(&G, xl, xu, dim, MC_calls / 5, r, s, &res_A2,
                                &err);
      // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
      // s->chisq);
    } while (fabs(s->chisq - 1.0) > MC_tol);

    par.n = 2;
    gsl_monte_vegas_init(s);
    gsl_monte_vegas_integrate(&G, xl, xu, dim, 10000, r, s, &res_A4, &err);
    // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
    // s->chisq);

    do {
      gsl_monte_vegas_integrate(&G, xl, xu, dim, MC_calls / 5, r, s, &res_A4,
                                &err);
      // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
      // s->chisq);
    } while (fabs(s->chisq - 1.0) > MC_tol);

    par.n = 3;
    gsl_monte_vegas_init(s);
    gsl_monte_vegas_integrate(&G, xl, xu, dim, 10000, r, s, &res_A6, &err);
    // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
    // s->chisq);

    do {
      gsl_monte_vegas_integrate(&G, xl, xu, dim, MC_calls / 5, r, s, &res_A6,
                                &err);
      // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
      // s->chisq);
    } while (fabs(s->chisq - 1.0) > MC_tol);

    A2[ik] = res_A2;
    A4[ik] = res_A4;
    A6[ik] = res_A6;
  }

  res["A2"] = A2;
  res["A4"] = A4;
  res["A6"] = A6;

  return res;
}

/* Bk222 + Bk321 kernel function for Monte-Carlo integration */
double direct_red::Aterm_Bk222_Bk321_kernel(double X[], size_t dim, void *param) {
  double jacobian, pi, eta, beta, k, kmin, kmax;
  double r, rmin, rmax, x, xmin, xmax;
  double q, mu, mumin, mumax, phi, phimin, phimax;
  double B1_tdd, B1_tdt, B1_ttd, B1_ttt, B2_tdd, B2_tdt, B2_ttd, B2_ttt;
  double kp, p;
  double res, integ;
  int n;
  Vector qq;
  Aterm_integral_params *par;

  pi = 4.0 * atan(1.0);

  par = (Aterm_integral_params *)param;
  k = par->k;
  eta = par->eta;
  beta = par->beta;
  n = par->n;
  kmin = par->kmin;
  kmax = par->kmax;
  mumin = par->mumin;
  mumax = par->mumax;
  phimin = par->phimin;
  phimax = par->phimax;

  rmin = kmin / k;
  rmax = kmax / k;
  r = exp(log(rmin) + (log(rmax) - log(rmin)) * X[0]);

  xmin = max(-1.0, (1.0 + sqr(r) - sqr(rmax)) / (2.0 * r));
  xmax = min(1.0, (1.0 + sqr(r) - sqr(rmin)) / (2.0 * r));
  if (r > 0.5)
    xmax = 0.5 / r;
  x = xmin + (xmax - xmin) * X[1];

  kp = k * sqrt(1.0 + sqr(r) - 2.0 * r * x);
  p = k * r;

  q = exp(log(kmin) + (log(kmax) - log(kmin)) * X[2]);
  mu = mumin + (mumax - mumin) * X[3];
  phi = phimin + (phimax - phimin) * X[4];

  qq.x = q * sqrt(1.0 - sqr(mu)) * cos(phi);
  qq.y = q * sqrt(1.0 - sqr(mu)) * sin(phi);
  qq.z = q * mu;

  B1_tdd = Bk222_Bk321_kernel(VELO, DENS, DENS, p, kp, k, qq, par);
  B1_tdt = Bk222_Bk321_kernel(VELO, DENS, VELO, p, kp, k, qq, par);
  B1_ttd = Bk222_Bk321_kernel(VELO, VELO, DENS, p, kp, k, qq, par);
  B1_ttt = Bk222_Bk321_kernel(VELO, VELO, VELO, p, kp, k, qq, par);
  B2_tdd = Bk222_Bk321_kernel(VELO, DENS, DENS, kp, p, k, qq, par);
  B2_tdt = Bk222_Bk321_kernel(VELO, DENS, VELO, kp, p, k, qq, par);
  B2_ttd = Bk222_Bk321_kernel(VELO, VELO, DENS, kp, p, k, qq, par);
  B2_ttt = Bk222_Bk321_kernel(VELO, VELO, VELO, kp, p, k, qq, par);

  integ = (beta * A_func(n, 1, 1, r, x) * B1_tdd +
           beta * At_func(n, 1, 1, r, x) * B2_tdd +
           sqr(beta) * A_func(n, 1, 2, r, x) * B1_tdt +
           sqr(beta) * At_func(n, 1, 2, r, x) * B2_tdt +
           sqr(beta) * A_func(n, 2, 1, r, x) * B1_ttd +
           sqr(beta) * At_func(n, 2, 1, r, x) * B2_ttd +
           cub(beta) * A_func(n, 2, 2, r, x) * B1_ttt +
           cub(beta) * At_func(n, 2, 2, r, x) * B2_ttt) *
          r * 2.0;

  jacobian = sqr(log(kmax) - log(kmin)) * (xmax - xmin) * (mumax - mumin) *
             (phimax - phimin);
  res = integ * cub(k) / sqr(2.0 * pi) * jacobian;

  return res;
}

/* B term correction term */
map<string, vector<double>> direct_red::Bterm(vector<double> k) {
  int nk;
  double rmin, rmax, xmin, xmax;
  double ki, ri, xi, wri, wxi, k1, k2;
  double P1_dt, P1_tt, P2_dt, P2_tt;
  double Bi[5];
  map<string, vector<double>> res;
  vector<double> B2, B4, B6, B8;

/*
          P1_dt = Pk1l_dt_spl(0.2);
          P1_tt = Pk1l_tt_spl(0.2);
          cout << "D dt:" << P1_dt << endl;
          cout << "D tt:" << P1_tt << endl;
*/

  nk = k.size();
  B2.resize(nk);
  B4.resize(nk);
  B6.resize(nk);
  B8.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];
    for (int n = 1; n <= 4; ++n)
      Bi[n] = 0.0;

    for (int ir = 0; ir < nr; ++ir) {
      rmin = kmin / ki;
      rmax = kmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wri, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wxi, t_x);
        k1 = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        k2 = ki * ri;

        if (flag_1loop) {
          P1_dt = spec->Plin(k1);
          P1_tt = spec->Plin(k1);
          P2_dt = spec->Plin(k2);
          P2_tt = spec->Plin(k2);
        } else {
          P1_dt = Pk1l_dt_spl(k1);
          P1_tt = Pk1l_tt_spl(k1);
          P2_dt = Pk1l_dt_spl(k2);
          P2_tt = Pk1l_tt_spl(k2);
        }

        for (int n = 1; n <= 4; ++n) {
          Bi[n] += wri * wxi *
                   (sqr(-beta) * B_func(n, 1, 1, ri, xi) * P1_dt * P2_dt /
                        (1.0 + sqr(ri) - 2.0 * ri * xi) +
                    cub(-beta) * B_func(n, 1, 2, ri, xi) * P1_dt * P2_tt /
                        (1.0 + sqr(ri) - 2.0 * ri * xi) +
                    cub(-beta) * B_func(n, 2, 1, ri, xi) * P1_tt * P2_dt /
                        sqr(1.0 + sqr(ri) - 2.0 * ri * xi) +
                    qua(-beta) * B_func(n, 2, 2, ri, xi) * P1_tt * P2_tt /
                        sqr(1.0 + sqr(ri) - 2.0 * ri * xi)) *
                   ri * 2.0;
        }
      }
    }
    B2[ik] = Bi[1] * cub(ki) / sqr(2.0 * pi);
    B4[ik] = Bi[2] * cub(ki) / sqr(2.0 * pi);
    B6[ik] = Bi[3] * cub(ki) / sqr(2.0 * pi);
    B8[ik] = Bi[4] * cub(ki) / sqr(2.0 * pi);
  }

  res["B2"] = B2;
  res["B4"] = B4;
  res["B6"] = B6;
  res["B8"] = B8;

  return res;
}

/* C term correction term */
map<string, vector<double>> direct_red::Cterm(vector<double> k) {
  int nk;
  double rmin, rmax, xmin, xmax;
  double ki, ri, xi, wri, wxi, k1, k2;
  double P1_dd, P1_dt, P1_tt, P2_dd, P2_dt, P2_tt;
  double Ci[5];
  map<string, vector<double>> res;
  vector<double> C2, C4, C6, C8;

  nk = k.size();
  C2.resize(nk);
  C4.resize(nk);
  C6.resize(nk);
  C8.resize(nk);

  for (int ik = 0; ik < nk; ++ik) {
    ki = k[ik];
    for (int n = 1; n <= 4; n++)
      Ci[n] = 0.0;

    for (int ir = 0; ir < nr; ++ir) {
      rmin = kmin / ki;
      rmax = kmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wri, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));
      if (ri > 0.5)
        xmax = 0.5 / ri;

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wxi, t_x);
        k1 = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);
        k2 = ki * ri;

        if (flag_1loop) {
          P1_dd = spec->Plin(k1);
          P1_dt = spec->Plin(k1);
          P1_tt = spec->Plin(k1);
          P2_dd = spec->Plin(k2);
          P2_dt = spec->Plin(k2);
          P2_tt = spec->Plin(k2);
        } else {
          P1_dd = Pk1l_dd_spl(k1);
          P1_dt = Pk1l_dt_spl(k1);
          P1_tt = Pk1l_tt_spl(k1);
          P2_dd = Pk1l_dd_spl(k2);
          P2_dt = Pk1l_dt_spl(k2);
          P2_tt = Pk1l_tt_spl(k2);
        }

        for (int n = 1; n <= 4; ++n) {
          Ci[n] += wri * wxi *
                   (sqr(-beta) * C_func(n, 1, 1, ri, xi) * P1_dd * P2_tt +
                    sqr(-beta) * Ct_func(n, 1, 1, ri, xi) * P1_tt * P2_dd +
                    cub(-beta) * C_func(n, 1, 2, ri, xi) * P1_dt * P2_tt +
                    cub(-beta) * Ct_func(n, 1, 2, ri, xi) * P1_tt * P2_dt +
                    cub(-beta) * C_func(n, 2, 1, ri, xi) * P1_dt * P2_tt +
                    cub(-beta) * Ct_func(n, 2, 1, ri, xi) * P1_tt * P2_dt +
                    qua(-beta) * C_func(n, 2, 2, ri, xi) * P1_tt * P2_tt +
                    qua(-beta) * Ct_func(n, 2, 2, ri, xi) * P1_tt * P2_tt) *
                   ri * 2.0;
        }
      }
    }
    C2[ik] = Ci[1] * cub(ki) / sqr(2.0 * pi);
    C4[ik] = Ci[2] * cub(ki) / sqr(2.0 * pi);
    C6[ik] = Ci[3] * cub(ki) / sqr(2.0 * pi);
    C8[ik] = Ci[4] * cub(ki) / sqr(2.0 * pi);
  }

  res["C2"] = C2;
  res["C4"] = C4;
  res["C6"] = C6;
  res["C8"] = C8;

  return res;
}

/* Bispectrum (1-loop) with IR-safe integral */
double direct_red::Bispec_1loop(Type a, Type b, Type c, double k1, double k2,
                                double k3) {
  double res, D, Bk222, Bk321, mu12, k12, k23, k31;
  double Bk211, G1reg_k1, G1reg_k2, G1reg_k3, G2reg_k1_k2, G2reg_k2_k3,
      G2reg_k3_k1;
  double F2_k1_k2, F2_k2_k3, F2_k3_k1;
  double G1_1l_k1, G1_1l_k2, G1_1l_k3, G2_1l_k1_k2, G2_1l_k2_k3, G2_1l_k3_k1;
  double sigmad2_k1, sigmad2_k2, sigmad2_k3, sigmad2_k12, sigmad2_k23,
      sigmad2_k31;
  double integ_Bk222, integ_Bk321, integ_mu_Bk222, integ_mu_Bk321, qi;
  double p1, p2, p3, r1, r2, r3;
  double Pk1, Pk2, Pk3, Pkq, Pkp1, Pkp2, Pkp3, Pkr1, Pkr2, Pkr3;
  Vector kk1, kk2, kk3, qq;
  Vector pp1, pp2, pp3, rr1, rr2, rr3;

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

  sigmad2_k1 = sigmad2_b(k1) * sqr(D);
  sigmad2_k2 = sigmad2_b(k2) * sqr(D);
  sigmad2_k3 = sigmad2_b(k3) * sqr(D);
  sigmad2_k12 = sigmad2_b(k12) * sqr(D);
  sigmad2_k23 = sigmad2_b(k23) * sqr(D);
  sigmad2_k31 = sigmad2_b(k31) * sqr(D);

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

/* Kernel function for Bk222+Bk321 (1-loop) with IR-safe integral */
double direct_red::Bk222_Bk321_kernel(Type a, Type b, Type c, double k1,
                                      double k2, double k3, Vector qq,
                                      Aterm_integral_params *par) {
  double res, D, pi, Bk222, Bk321, mu12, k12, k23, k31;
  double sigmad2_k1, sigmad2_k2, sigmad2_k3, sigmad2_k12, sigmad2_k23,
      sigmad2_k31;
  double qi, p1, p2, p3, r1, r2, r3;
  double Pk1, Pk2, Pk3, Pkq, Pkp1, Pkp2, Pkp3, Pkr1, Pkr2, Pkr3;
  bool flag_SPT;
  Vector kk1, kk2, kk3;
  Vector pp1, pp2, pp3, rr1, rr2, rr3;

  pi = 4.0 * atan(1.0);
  D = exp(par->eta);
  flag_SPT = par->flag_SPT;

  mu12 = -(sqr(k1) + sqr(k2) - sqr(k3)) / (2.0 * k1 * k2);

  kk1.x = 0.0, kk1.y = 0.0, kk1.z = k1;
  kk2.x = 0.0, kk2.y = k2 * sqrt(1.0 - sqr(mu12)), kk2.z = k2 * mu12;
  kk3 = -kk1 - kk2;

  Pk1 = par->spec->P0(k1);
  Pk2 = par->spec->P0(k2);
  Pk3 = par->spec->P0(k3);

  k12 = sqrt((kk1 + kk2) * (kk1 + kk2));
  k23 = sqrt((kk2 + kk3) * (kk2 + kk3));
  k31 = sqrt((kk3 + kk1) * (kk3 + kk1));

  sigmad2_k1 =
      gsl_spline_eval(par->spl_sigmad2, log(k1), par->acc_sigmad2) * sqr(D);
  sigmad2_k2 =
      gsl_spline_eval(par->spl_sigmad2, log(k2), par->acc_sigmad2) * sqr(D);
  sigmad2_k3 =
      gsl_spline_eval(par->spl_sigmad2, log(k3), par->acc_sigmad2) * sqr(D);
  sigmad2_k12 =
      gsl_spline_eval(par->spl_sigmad2, log(k12), par->acc_sigmad2) * sqr(D);
  sigmad2_k23 =
      gsl_spline_eval(par->spl_sigmad2, log(k23), par->acc_sigmad2) * sqr(D);
  sigmad2_k31 =
      gsl_spline_eval(par->spl_sigmad2, log(k31), par->acc_sigmad2) * sqr(D);

  Bk222 = 0.0;
  Bk321 = 0.0;
  qi = sqrt(qq * qq);
  Pkq = par->spec->P0(qi);

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

  Pkp1 = par->spec->P0(p1);
  Pkp2 = par->spec->P0(p2);
  Pkp3 = par->spec->P0(p3);
  Pkr1 = par->spec->P0(r1);
  Pkr2 = par->spec->P0(r2);
  Pkr3 = par->spec->P0(r3);

  if (flag_SPT) {
    // IR-safe integrand for Bk222 with SPT
    if (p1 > qi && r2 > qi)
      Bk222 += F2_sym(a, pp1, qq) * F2_sym(b, rr2, -qq) *
               F2_sym(c, -rr2, -pp1) * Pkp1 * Pkq * Pkr2;

    if (r1 > qi && p2 > qi)
      Bk222 += F2_sym(a, rr1, -qq) * F2_sym(b, pp2, qq) *
               F2_sym(c, -pp2, -rr1) * Pkr1 * Pkq * Pkp2;

    if (p3 > qi && r2 > qi)
      Bk222 += F2_sym(c, pp3, qq) * F2_sym(b, rr2, -qq) *
               F2_sym(a, -rr2, -pp3) * Pkp3 * Pkq * Pkr2;

    if (r3 > qi && p2 > qi)
      Bk222 += F2_sym(c, rr3, -qq) * F2_sym(b, pp2, qq) *
               F2_sym(a, -pp2, -rr3) * Pkr3 * Pkq * Pkp2;

    if (p1 > qi && r3 > qi)
      Bk222 += F2_sym(a, pp1, qq) * F2_sym(c, rr3, -qq) *
               F2_sym(b, -rr3, -pp1) * Pkp1 * Pkq * Pkr3;

    if (r1 > qi && p3 > qi)
      Bk222 += F2_sym(a, rr1, -qq) * F2_sym(c, pp3, qq) *
               F2_sym(b, -pp3, -rr1) * Pkr1 * Pkq * Pkp3;

    // IR-safe integrand for Bk321 with SPT
    if (p2 > qi)
      Bk321 +=
          F3_sym(a, -kk3, -pp2, -qq) * F2_sym(b, pp2, qq) * Pkp2 * Pkq * Pk3 +
          F3_sym(c, -kk1, -pp2, -qq) * F2_sym(b, pp2, qq) * Pkp2 * Pkq * Pk1;

    if (r2 > qi)
      Bk321 +=
          F3_sym(a, -kk3, -rr2, qq) * F2_sym(b, rr2, -qq) * Pkr2 * Pkq * Pk3 +
          F3_sym(c, -kk1, -rr2, qq) * F2_sym(b, rr2, -qq) * Pkr2 * Pkq * Pk1;

    if (p3 > qi)
      Bk321 +=
          F3_sym(a, -kk2, -pp3, -qq) * F2_sym(c, pp3, qq) * Pkp3 * Pkq * Pk2 +
          F3_sym(b, -kk1, -pp3, -qq) * F2_sym(c, pp3, qq) * Pkp3 * Pkq * Pk1;

    if (r3 > qi)
      Bk321 +=
          F3_sym(a, -kk2, -rr3, qq) * F2_sym(c, rr3, -qq) * Pkr3 * Pkq * Pk2 +
          F3_sym(b, -kk1, -rr3, qq) * F2_sym(c, rr3, -qq) * Pkr3 * Pkq * Pk1;

    if (p1 > qi)
      Bk321 +=
          F3_sym(b, -kk3, -pp1, -qq) * F2_sym(a, pp1, qq) * Pkp1 * Pkq * Pk3 +
          F3_sym(c, -kk2, -pp1, -qq) * F2_sym(a, pp1, qq) * Pkp1 * Pkq * Pk2;

    if (r1 > qi)
      Bk321 +=
          F3_sym(b, -kk3, -rr1, qq) * F2_sym(a, rr1, -qq) * Pkr1 * Pkq * Pk3 +
          F3_sym(c, -kk2, -rr1, qq) * F2_sym(a, rr1, -qq) * Pkr1 * Pkq * Pk2;
  } else {
    // IR-safe integrand for Bk222 with RegPT
    if (p1 > qi && r2 > qi)
      Bk222 += F2_sym(a, pp1, qq) * F2_sym(b, rr2, -qq) *
               F2_sym(c, -rr2, -pp1) * Pkp1 * Pkq * Pkr2 *
               exp(-0.5 * (sqr(k1) * sigmad2_k1 + sqr(k2) * sigmad2_k2 +
                           sqr(k12) * sigmad2_k12));

    if (r1 > qi && p2 > qi)
      Bk222 += F2_sym(a, rr1, -qq) * F2_sym(b, pp2, qq) *
               F2_sym(c, -pp2, -rr1) * Pkr1 * Pkq * Pkp2 *
               exp(-0.5 * (sqr(k1) * sigmad2_k1 + sqr(k2) * sigmad2_k2 +
                           sqr(k12) * sigmad2_k12));

    if (p3 > qi && r2 > qi)
      Bk222 += F2_sym(c, pp3, qq) * F2_sym(b, rr2, -qq) *
               F2_sym(a, -rr2, -pp3) * Pkp3 * Pkq * Pkr2 *
               exp(-0.5 * (sqr(k2) * sigmad2_k2 + sqr(k3) * sigmad2_k3 +
                           sqr(k23) * sigmad2_k23));

    if (r3 > qi && p2 > qi)
      Bk222 += F2_sym(c, rr3, -qq) * F2_sym(b, pp2, qq) *
               F2_sym(a, -pp2, -rr3) * Pkr3 * Pkq * Pkp2 *
               exp(-0.5 * (sqr(k2) * sigmad2_k2 + sqr(k3) * sigmad2_k3 +
                           sqr(k23) * sigmad2_k23));

    if (p1 > qi && r3 > qi)
      Bk222 += F2_sym(a, pp1, qq) * F2_sym(c, rr3, -qq) *
               F2_sym(b, -rr3, -pp1) * Pkp1 * Pkq * Pkr3 *
               exp(-0.5 * (sqr(k3) * sigmad2_k3 + sqr(k1) * sigmad2_k1 +
                           sqr(k31) * sigmad2_k31));

    if (r1 > qi && p3 > qi)
      Bk222 += F2_sym(a, rr1, -qq) * F2_sym(c, pp3, qq) *
               F2_sym(b, -pp3, -rr1) * Pkr1 * Pkq * Pkp3 *
               exp(-0.5 * (sqr(k3) * sigmad2_k3 + sqr(k1) * sigmad2_k1 +
                           sqr(k31) * sigmad2_k31));

    // IR-safe integrand for Bk321 with RegPT
    if (p2 > qi)
      Bk321 +=
          F3_sym(a, -kk3, -pp2, -qq) * F2_sym(b, pp2, qq) * Pkp2 * Pkq * Pk3 *
              exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                          sqr(k3) * sigmad2_k3)) +
          F3_sym(c, -kk1, -pp2, -qq) * F2_sym(b, pp2, qq) * Pkp2 * Pkq * Pk1 *
              exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                          sqr(k2) * sigmad2_k2));

    if (r2 > qi)
      Bk321 +=
          F3_sym(a, -kk3, -rr2, qq) * F2_sym(b, rr2, -qq) * Pkr2 * Pkq * Pk3 *
              exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                          sqr(k3) * sigmad2_k3)) +
          F3_sym(c, -kk1, -rr2, qq) * F2_sym(b, rr2, -qq) * Pkr2 * Pkq * Pk1 *
              exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                          sqr(k2) * sigmad2_k2));

    if (p3 > qi)
      Bk321 +=
          F3_sym(a, -kk2, -pp3, -qq) * F2_sym(c, pp3, qq) * Pkp3 * Pkq * Pk2 *
              exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                          sqr(k3) * sigmad2_k3)) +
          F3_sym(b, -kk1, -pp3, -qq) * F2_sym(c, pp3, qq) * Pkp3 * Pkq * Pk1 *
              exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                          sqr(k1) * sigmad2_k1));

    if (r3 > qi)
      Bk321 +=
          F3_sym(a, -kk2, -rr3, qq) * F2_sym(c, rr3, -qq) * Pkr3 * Pkq * Pk2 *
              exp(-0.5 * (sqr(k23) * sigmad2_k23 + sqr(k2) * sigmad2_k2 +
                          sqr(k3) * sigmad2_k3)) +
          F3_sym(b, -kk1, -rr3, qq) * F2_sym(c, rr3, -qq) * Pkr3 * Pkq * Pk1 *
              exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                          sqr(k1) * sigmad2_k1));

    if (p1 > qi)
      Bk321 +=
          F3_sym(b, -kk3, -pp1, -qq) * F2_sym(a, pp1, qq) * Pkp1 * Pkq * Pk3 *
              exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                          sqr(k1) * sigmad2_k1)) +
          F3_sym(c, -kk2, -pp1, -qq) * F2_sym(a, pp1, qq) * Pkp1 * Pkq * Pk2 *
              exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                          sqr(k2) * sigmad2_k2));

    if (r1 > qi)
      Bk321 +=
          F3_sym(b, -kk3, -rr1, qq) * F2_sym(a, rr1, -qq) * Pkr1 * Pkq * Pk3 *
              exp(-0.5 * (sqr(k31) * sigmad2_k31 + sqr(k3) * sigmad2_k3 +
                          sqr(k1) * sigmad2_k1)) +
          F3_sym(c, -kk2, -rr1, qq) * F2_sym(a, rr1, -qq) * Pkr1 * Pkq * Pk2 *
              exp(-0.5 * (sqr(k12) * sigmad2_k12 + sqr(k1) * sigmad2_k1 +
                          sqr(k2) * sigmad2_k2));
  }

  Bk222 *= 8.0 / cub(2.0 * pi) * cub(qi) / 2.0;
  Bk321 *= 6.0 / cub(2.0 * pi) * cub(qi);

  res = sqr(D) * qua(D) * (Bk222 + Bk321);

  return res;
}

/* Bk211 (1-loop) with IR-safe integral */
double direct_red::Bk211(Type a, Type b, Type c, double k1, double k2,
                         double k3) {
  double res, D, mu12, k12, k23, k31;
  double Pk1, Pk2, Pk3;
  double sigmad2_k1, sigmad2_k2, sigmad2_k3, sigmad2_k12, sigmad2_k23,
      sigmad2_k31;
  double Bk211, G1reg_k1, G1reg_k2, G1reg_k3, G2reg_k1_k2, G2reg_k2_k3,
      G2reg_k3_k1;
  double F2_k1_k2, F2_k2_k3, F2_k3_k1;
  double G1_1l_k1, G1_1l_k2, G1_1l_k3, G2_1l_k1_k2, G2_1l_k2_k3, G2_1l_k3_k1;
  Vector kk1, kk2, kk3;

  D = exp(eta);
  mu12 = -(sqr(k1) + sqr(k2) - sqr(k3)) / (2.0 * k1 * k2);

  kk1.x = 0.0, kk1.y = 0.0, kk1.z = k1;
  kk2.x = 0.0, kk2.y = k2 * sqrt(1.0 - sqr(mu12)), kk2.z = k2 * mu12;
  kk3 = -kk1 - kk2;

  k12 = sqrt((kk1 + kk2) * (kk1 + kk2));
  k23 = sqrt((kk2 + kk3) * (kk2 + kk3));
  k31 = sqrt((kk3 + kk1) * (kk3 + kk1));

  Pk1 = spec->P0(k1);
  Pk2 = spec->P0(k2);
  Pk3 = spec->P0(k3);

  sigmad2_k1 = sigmad2_b(k1) * sqr(D);
  sigmad2_k2 = sigmad2_b(k2) * sqr(D);
  sigmad2_k3 = sigmad2_b(k3) * sqr(D);
  sigmad2_k12 = sigmad2_b(k12) * sqr(D);
  sigmad2_k23 = sigmad2_b(k23) * sqr(D);
  sigmad2_k31 = sigmad2_b(k31) * sqr(D);

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

  res = qua(D) * Bk211;

  return res;
}

/* Bispectrum (tree level) */
double direct_red::Bispec_tree(Type a, Type b, Type c, double k1, double k2,
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

  sigmad2_k1 = sigmad2_b(k1) * sqr(D);
  sigmad2_k2 = sigmad2_b(k2) * sqr(D);
  sigmad2_k3 = sigmad2_b(k3) * sqr(D);
  sigmad2_k12 = sigmad2_b(k12) * sqr(D);
  sigmad2_k23 = sigmad2_b(k23) * sqr(D);
  sigmad2_k31 = sigmad2_b(k31) * sqr(D);

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

/* auxiliary functions for A term */
double A_func(int a, int b, int c, double r, double x) {
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

double At_func(int a, int b, int c, double r, double x) {
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
double B_func(int a, int b, int c, double r, double x) {
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

/* auxiliary functions for C term */
double C_func(int a, int b, int c, double r, double x) {
  if (a == 1 && b == 1 && c == 1) {
    return -(x * x - 1.0) / 4.0;
  } else if (a == 1 && b == 1 && c == 2) {
    return -3.0 * r * r * sqr(x * x - 1.0) /
           (8.0 * (1.0 + r * r - 2.0 * r * x));
  } else if (a == 1 && b == 2 && c == 2) {
    return -5.0 * qua(r) * cub(x * x - 1.0) /
           (16.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 2 && b == 1 && c == 1) {
    return (3.0 * x * x - 1.0) / 4.0;
  } else if (a == 2 && b == 1 && c == 2) {
    return (x * x - 1.0) *
           (2.0 - 12.0 * r * x - 3 * r * r + 15.0 * r * r * x * x) /
           (4.0 * (1.0 + r * r - 2.0 * r * x));
  } else if (a == 2 && b == 2 && c == 2) {
    return 3.0 * r * r * sqr(x * x - 1.0) *
           (7.0 - 30.0 * r * x - 5.0 * r * r + 35.0 * r * r * x * x) /
           (16.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 3 && b == 1 && c == 2) {
    return -1.0 *
           (-4.0 + 12.0 * x * x + 8.0 * r * x * (3.0 - 5 * x * x) +
            r * r * (3.0 - 30.0 * x * x + 35.0 * qua(x))) /
           (8.0 * (1.0 + r * r - 2.0 * r * x));
  } else if (a == 3 && b == 2 && c == 2) {
    return -1.0 * (x * x - 1.0) *
           (4.0 +
            3.0 * r *
                (-16.0 * x +
                 r * (-14.0 + 70.0 * x * x +
                      20.0 * r * x * (3.0 - 7.0 * x * x) +
                      5.0 * r * r * (1.0 - 14.0 * x * x + 21.0 * qua(x))))) /
           (16.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 4 && b == 2 && c == 2) {
    return (-4.0 + 12 * x * x + 16.0 * r * x * (3.0 - 5.0 * x * x) +
            7.0 * r * r * (3.0 - 30.0 * x * x + 35.0 * qua(x)) -
            6.0 * cub(r) * x * (15.0 - 70.0 * x * x + 63.0 * qua(x))) /
               (16.0 * sqr(1.0 + r * r - 2.0 * r * x)) +
           qua(r) *
               (-5.0 + 21.0 * x * x * (5.0 - 15.0 * x * x + 11.0 * qua(x))) /
               (16.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else
    return 0.0;
}

double Ct_func(int a, int b, int c, double r, double x) {
  if (a == 1 && b == 1 && c == 1) {
    return -qua(r) * (x * x - 1.0) / (4.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 1 && b == 1 && c == 2) {
    return -3.0 * qua(r) * sqr(x * x - 1.0) /
           (8.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 2 && b == 1 && c == 1) {
    return r * r * (2.0 - 4.0 * r * x - r * r + 3.0 * r * r * x * x) /
           (4.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 2 && b == 1 && c == 2) {
    return r * r * (x * x - 1.0) *
           (2.0 - 12.0 * r * x - 3.0 * r * r + 15.0 * r * r * x * x) /
           (4.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else if (a == 3 && b == 1 && c == 2) {
    return -r * r *
           (-4.0 + 12.0 * x * x + 8.0 * r * x * (3.0 - 5.0 * x * x) +
            r * r * (3.0 - 30.0 * x * x + 35.0 * qua(x))) /
           (8.0 * sqr(1.0 + r * r - 2.0 * r * x));
  } else
    return 0.0;
}
