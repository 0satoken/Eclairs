#include "IR_EFT.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

IR_EFT::IR_EFT(params &para, cosmology &cosmo, spectra &spec) {
  this->para = &para;
  this->cosmo = &cosmo;

  pi = 4.0 * atan(1.0);
  eta = cosmo.get_eta();
  f = cosmo.get_growth_rate();

  /* bias parameters */
  b1 = para.dparams["b1"];
  beta = f / b1;

  /* AP parameters */
  flag_AP = para.bparams["AP"];
  alpha_perp = para.dparams["alpha_perp"];
  alpha_para = para.dparams["alpha_para"];
  rs_drag_ratio = para.dparams["rs_drag_ratio"];

  /* counter terms */
  c0 = para.dparams["IREFT_c0"];
  c2 = para.dparams["IREFT_c2"];
  c4 = para.dparams["IREFT_c4"];
  cd4 = para.dparams["IREFT_cd4"];
  Pshot = para.dparams["IREFT_Pshot"];
  kS = para.dparams["IREFT_kS"];
  rs = para.dparams["IREFT_rs"];

  /* parameters for smoothing spectrum */
  nk_spl = para.iparams["nk_spl"];
  nsm = para.iparams["smooth_nk"];
  ksmmin = para.dparams["smooth_kmin"];
  ksmmax = para.dparams["smooth_kmax"];
  lambda = para.dparams["smooth_lambda"];

  /* setup for numerical integration */
  kmin = para.dparams["kmin"];
  kmax = para.dparams["kmax"];
  nr = para.iparams["IREFT_nr"];
  nx = para.iparams["IREFT_nx"];
  nq = para.iparams["IREFT_nq"];
  nmuint = para.iparams["IREFT_nmu"];

  /* setup for numerical integration */
  L = para.dparams["grid_L"];
  ng = para.iparams["grid_ng"];

  t_r = gsl_integration_glfixed_table_alloc(nr);
  t_x = gsl_integration_glfixed_table_alloc(nx);
  t_mu = gsl_integration_glfixed_table_alloc(nmuint);

  acc_Pk_1l_nw = new gsl_interp_accel *[5];
  acc_Pk_1l_w = new gsl_interp_accel *[5];

  spl_Pk_1l_nw = new gsl_spline *[5];
  spl_Pk_1l_w = new gsl_spline *[5];

  set_smoothed_spectra();
  calc_Sigma();
  construct_spline_Pk();
}

IR_EFT::~IR_EFT() {
  /* freeing memories */
  gsl_integration_glfixed_table_free(t_r);
  gsl_integration_glfixed_table_free(t_x);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_spline_free(Pspl_w);
  gsl_spline_free(Pspl_nw);
  gsl_interp_accel_free(acc_w);
  gsl_interp_accel_free(acc_nw);

  for (int i = 0; i < 5; ++i) {
    gsl_spline_free(spl_Pk_1l_nw[i]);
    gsl_spline_free(spl_Pk_1l_w[i]);
    gsl_interp_accel_free(acc_Pk_1l_nw[i]);
    gsl_interp_accel_free(acc_Pk_1l_w[i]);
  }

  delete[] spl_Pk_1l_nw;
  delete[] spl_Pk_1l_w;
  delete[] acc_Pk_1l_nw;
  delete[] acc_Pk_1l_w;
}

double IR_EFT::P_w(double k) {
  return exp(2.0 * eta) * gsl_spline_eval(Pspl_w, k, acc_w);
}

double IR_EFT::P_nw(double k) {
  return exp(2.0 * eta) * gsl_spline_eval(Pspl_nw, k, acc_nw);
}

/*
 * Smoothes wiggle feature with Gaussian smoothing in log space.
 * For details, see Appedix A of Vlah et al., JCAP, 03(2016)057
 */
void IR_EFT::set_smoothed_spectra(void) {
  double klog, qlogi, qi, res;
  double *k_, *P_w_, *P_nw_;
  double *qlog, *wqlog;
  gsl_integration_glfixed_table *t_sm;

  t_sm = gsl_integration_glfixed_table_alloc(nsm);
  qlog = new double[nsm];
  wqlog = new double[nsm];

  k_ = new double[nk_spl];
  P_w_ = new double[nk_spl];
  P_nw_ = new double[nk_spl];

  for (int i = 0; i < nk_spl; ++i) {
    k_[i] = (log(kmax) - log(kmin)) / (double(nk_spl) - 1.0) * i + log(kmin);
    k_[i] = exp(k_[i]);
  }

  for (int i = 0; i < nsm; ++i) {
    gsl_integration_glfixed_point(log10(ksmmin), log10(ksmmax), i, &qlog[i], &wqlog[i], t_sm);
  }

  for (int j = 0; j < nk_spl; j++) {
    klog = log10(k_[j]);

    res = 0.0;
    for (int i = 0; i < nsm; i++) {
      qi = pow(10.0, qlog[i]);
      qlogi = qlog[i];
      res += wqlog[i] * cosmo->P0(qi) / cosmo->Pno_wiggle0(qi) *
             exp(-0.5 / sqr(lambda) * sqr(klog - qlogi));
    }
    res *= 1.0 / sqrt(2.0 * pi) / lambda;
    res *= cosmo->Pno_wiggle0(k_[j]);

    P_nw_[j] = res;
    P_w_[j] = cosmo->P0(k_[j]) - res;
  }

  /* spline function for linear power spectrum */
  acc_w = gsl_interp_accel_alloc();
  Pspl_w = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  gsl_spline_init(Pspl_w, k_, P_w_, nk_spl);

  /* spline function for no-wiggle power spectrum */
  acc_nw = gsl_interp_accel_alloc();
  Pspl_nw = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  gsl_spline_init(Pspl_nw, k_, P_nw_, nk_spl);

  delete[] k_;
  delete[] P_w_;
  delete[] P_nw_;
  delete[] wqlog;
  delete[] qlog;
  gsl_integration_glfixed_table_free(t_sm);

  return;
}


/* calculate and set damping factors (Sigma^2 and delta Sigma^2) */
void IR_EFT::calc_Sigma(void) {
  double qi, wi, Sigma2_, deltaSigma2_;
  gsl_integration_glfixed_table *t_q;

  t_q = gsl_integration_glfixed_table_alloc(nq);

  Sigma2_ = 0.0;
  deltaSigma2_ = 0.0;
  for (int i = 0; i < nq; ++i) {
    gsl_integration_glfixed_point(log(kmin), log10(kS), i, &qi, &wi, t_q);
    qi = exp(qi);

    Sigma2_ +=
        wi * qi * P_nw(qi) *
        (1.0 - gsl_sf_bessel_j0(qi * rs) + 2.0 * gsl_sf_bessel_j2(qi * rs));
    deltaSigma2_ += wi * qi * P_nw(qi) * gsl_sf_bessel_j2(qi * rs);
  }

  Sigma2_ = 1.0 / (6.0 * sqr(pi)) * Sigma2_;
  deltaSigma2_ = 1.0 / (2.0 * sqr(pi)) * pi * deltaSigma2_;

  Sigma2 = Sigma2_;
  deltaSigma2 = deltaSigma2_;

  gsl_integration_glfixed_table_free(t_q);

  return;
}

void IR_EFT::construct_spline_Pk(void) {
  double ki, mui, Pkmui, ri, xi, wr, wx, qi, kqi;
  double rmin, rmax, xmin, xmax;
  double Pk, Pk_w, Pk_nw, Pq, Pq_nw, Pkq, Pkq_nw;
  double integ_13[3], integ_22[5], integ_13_nw[3], integ_22_nw[5];
  double *k, *Pk_1l, *Pk_1l_w, *Pk_1l_nw;

  for (int i = 0; i < 5; ++i) {
    acc_Pk_1l_nw[i] = gsl_interp_accel_alloc();
    acc_Pk_1l_w[i] = gsl_interp_accel_alloc();
    spl_Pk_1l_nw[i] = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
    spl_Pk_1l_w[i] = gsl_spline_alloc(gsl_interp_cspline, nk_spl);
  }

  k = new double[nk_spl];
  Pk_1l = new double[5 * nk_spl];
  Pk_1l_w = new double[5 * nk_spl];
  Pk_1l_nw = new double[5 * nk_spl];

  for (int ik = 0; ik < nk_spl; ++ik) {
    k[ik] = (log(kmax) - log(kmin)) / (double(nk_spl) - 0.0) * ik + log(kmin);
    k[ik] = exp(k[ik]);
    ki = k[ik];

    Pk = cosmo->Plin(ki);
    Pk_nw = P_nw(ki);
    Pk_w = P_w(ki);

    for (int n = 0; n < 3; ++n) {
      integ_13[n] = 0.0;
      integ_13_nw[n] = 0.0;
    }

    for (int n = 0; n < 5; ++n) {
      integ_22[n] = 0.0;
      integ_22_nw[n] = 0.0;
    }

    for (int ir = 0; ir < nr; ir++) {
      rmin = kmin / ki;
      rmax = kmax / ki;
      gsl_integration_glfixed_point(log(rmin), log(rmax), ir, &ri, &wr, t_r);
      ri = exp(ri);

      xmin = max(-1.0, (1.0 + sqr(ri) - sqr(rmax)) / (2.0 * ri));
      xmax = min(1.0, (1.0 + sqr(ri) - sqr(rmin)) / (2.0 * ri));

      qi = ki * ri;
      Pq = cosmo->Plin(qi);
      Pq_nw = P_nw(qi);

      for (int n = 0; n <= 2; ++n) {
        for (int m = 0; m <= 3; ++m) {
          integ_13[n] += pow(beta, (double)m) * wr * ri * Pq * B_func(n, m, ri);
          integ_13_nw[n] +=
              pow(beta, (double)m) * wr * ri * Pq_nw * B_func(n, m, ri);
        }
      }

      for (int ix = 0; ix < nx; ++ix) {
        gsl_integration_glfixed_point(xmin, xmax, ix, &xi, &wx, t_x);
        kqi = ki * sqrt(1.0 + sqr(ri) - 2.0 * ri * xi);

        Pkq = cosmo->Plin(kqi);
        Pkq_nw = P_nw(kqi);

        for (int n = 0; n <= 4; ++n) {
          for (int m = 0; m <= 4; ++m) {
            integ_22[n] += pow(beta, (double)m) * wr * wx * ri * Pq * Pkq *
                           A_func(n, m, ri, xi) /
                           sqr(1.0 + sqr(ri) - 2.0 * ri * xi);
            integ_22_nw[n] += pow(beta, (double)m) * wr * wx * ri * Pq_nw *
                              Pkq_nw * A_func(n, m, ri, xi) /
                              sqr(1.0 + sqr(ri) - 2.0 * ri * xi);
          }
        }
      }
    }

    for (int n = 0; n < 3; ++n) {
      integ_13[n] *= qua(b1) * Pk * cub(ki) / (4.0 * sqr(pi));
      integ_13_nw[n] *= qua(b1) * Pk_nw * cub(ki) / (4.0 * sqr(pi));
    }

    for (int n = 0; n < 5; ++n) {
      integ_22[n] *= qua(b1) * cub(ki) / (4.0 * sqr(pi));
      integ_22_nw[n] *= qua(b1) * cub(ki) / (4.0 * sqr(pi));
    }

    Pk_1l[ik + nk_spl * 0] = integ_22[0] + integ_13[0];
    Pk_1l[ik + nk_spl * 1] = integ_22[1] + integ_13[1] + beta * integ_13[0];
    Pk_1l[ik + nk_spl * 2] = integ_22[2] + integ_13[2] + beta * integ_13[1];
    Pk_1l[ik + nk_spl * 3] = integ_22[3] + beta * integ_13[2];
    Pk_1l[ik + nk_spl * 4] = integ_22[4];

    Pk_1l_nw[ik + nk_spl * 0] = integ_22_nw[0] + integ_13_nw[0];
    Pk_1l_nw[ik + nk_spl * 1] =
        integ_22_nw[1] + integ_13_nw[1] + beta * integ_13_nw[0];
    Pk_1l_nw[ik + nk_spl * 2] =
        integ_22_nw[2] + integ_13_nw[2] + beta * integ_13_nw[1];
    Pk_1l_nw[ik + nk_spl * 3] = integ_22_nw[3] + beta * integ_13_nw[2];
    Pk_1l_nw[ik + nk_spl * 4] = integ_22_nw[4];

    for (int n = 0; n < 5; ++n) {
      Pk_1l_w[ik + nk_spl * n] =
          Pk_1l[ik + nk_spl * n] - Pk_1l_nw[ik + nk_spl * n];
    }

    // printf("%g %g %g %g %g %g\n", k[ik], Pk_1l[ik+nk_spl*0],
    // Pk_1l[ik+nk_spl*1], Pk_1l[ik+nk_spl*2], Pk_1l[ik+nk_spl*3],
    // Pk_1l[ik+nk_spl*4]);
  }

  for (int i = 0; i < 5; ++i) {
    gsl_spline_init(spl_Pk_1l_nw[i], k, &Pk_1l_nw[i * nk_spl], nk_spl);
    gsl_spline_init(spl_Pk_1l_w[i], k, &Pk_1l_w[i * nk_spl], nk_spl);
  }

  delete[] Pk_1l;
  delete[] Pk_1l_nw;
  delete[] Pk_1l_w;

  return;
}

vector<vector<double>> IR_EFT::calc_Pkmu_1l(vector<double> k,
                                            vector<double> mu) {
  int nk, nmu;
  double ki, mui, Pkmui, ri, xi, wr, wx, qi, kqi;
  double rmin, rmax, xmin, xmax;
  double Pk, Pk_w, Pk_nw, Pq, Pq_nw, Pkq, Pkq_nw;
  double Sigma2_tot, Pk_lin_nw, Pk_lin_w, Pk_1l_nw, Pk_1l_w, Pnw_ctr, Pw_ctr;
  double integ_13[3], integ_22[5], integ_13_nw[3], integ_22_nw[5];
  double Pkmu_1l[5], Pkmu_1l_w[5], Pkmu_1l_nw[5];

  nk = k.size();
  nmu = mu.size();

  vector<vector<double>> res(nk, vector<double>(nmu));

  for (int ik = 0; ik < nk; ++ik) {
    for (int imu = 0; imu < nmu; ++imu) {
      if (flag_AP) {
        ki = k[ik] / alpha_perp *
             sqrt(1.0 + sqr(mu[imu]) * (sqr(alpha_perp / alpha_para) - 1.0));
        mui = mu[imu] * alpha_perp / alpha_para /
              sqrt(1.0 + sqr(mu[imu]) * (sqr(alpha_perp / alpha_para) - 1.0));
      } else {
        ki = k[ik];
        mui = mu[imu];
      }

      Pk = cosmo->Plin(ki);
      Pk_nw = P_nw(ki);
      Pk_w = P_w(ki);

      Sigma2_tot = (1.0 + f * sqr(mui) * (2.0 + f)) * Sigma2 +
                   sqr(f) * sqr(mui) * (sqr(mui) - 1.0) * deltaSigma2;

      Pk_lin_nw = sqr(1.0 + beta * sqr(mui)) * sqr(b1) * Pk_nw;
      Pk_lin_w = sqr(1.0 + beta * sqr(mui)) * sqr(b1) * Pk_w;

      if (ki < kmin || ki > kmax) {
        Pk_1l_nw = 0.0;
        Pk_1l_w = 0.0;
      } else {
        Pk_1l_nw = 0.0;
        Pk_1l_w = 0.0;
        for (int n = 0; n < 5; ++n) {
          Pk_1l_nw += pow(mui, 2.0 * n) *
                      gsl_spline_eval(spl_Pk_1l_nw[n], ki, acc_Pk_1l_nw[n]);
          Pk_1l_w += pow(mui, 2.0 * n) *
                     gsl_spline_eval(spl_Pk_1l_w[n], ki, acc_Pk_1l_w[n]);
        }
      }

      Pnw_ctr = -2.0 * c0 * sqr(ki) * Pk_lin_nw -
                2.0 * c2 * f * sqr(mui) * sqr(ki) * Pk_lin_nw -
                2.0 * c4 * sqr(f) * qua(mui) * sqr(ki) * Pk_lin_nw +
                cd4 * qua(f) * qua(mui) * qua(ki) * sqr(1.0 + beta * sqr(mui)) *
                    sqr(b1) * Pk_lin_nw;
      Pw_ctr = -2.0 * c0 * sqr(ki) * Pk_lin_w -
               2.0 * c2 * f * sqr(mui) * sqr(ki) * Pk_lin_w -
               2.0 * c4 * sqr(f) * qua(mui) * sqr(ki) * Pk_lin_w +
               cd4 * qua(f) * qua(mui) * qua(ki) * sqr(1.0 + beta * sqr(mui)) *
                   sqr(b1) * Pk_lin_w;

      Pkmui = Pk_lin_nw + Pk_1l_nw + Pnw_ctr +
              exp(-sqr(ki) * Sigma2_tot) *
                  (Pk_lin_w * (1.0 + sqr(ki) * Sigma2_tot) + Pk_1l_w + Pw_ctr) +
              Pshot;

      if (flag_AP) {
        Pkmui /= cub(rs_drag_ratio) * sqr(alpha_perp) * alpha_para;
      }

      res[ik][imu] = Pkmui;
    }
  }

  return res;
}

double IR_EFT::calc_Pkmu_1l(double k, double mu) {
  double ki, mui, Pkmui, ri, xi, wr, wx, qi, kqi;
  double rmin, rmax, xmin, xmax;
  double Pk, Pk_w, Pk_nw, Pq, Pq_nw, Pkq, Pkq_nw;
  double Sigma2_tot, Pk_lin_nw, Pk_lin_w, Pk_1l_nw, Pk_1l_w, Pnw_ctr, Pw_ctr;
  double integ_13[3], integ_22[5], integ_13_nw[3], integ_22_nw[5];
  double Pkmu_1l[5], Pkmu_1l_w[5], Pkmu_1l_nw[5];

  if (flag_AP) {
    ki = k / alpha_perp *
         sqrt(1.0 + sqr(mu) * (sqr(alpha_perp / alpha_para) - 1.0));
    mui = mu * alpha_perp / alpha_para /
          sqrt(1.0 + sqr(mu) * (sqr(alpha_perp / alpha_para) - 1.0));
  } else {
    ki = k;
    mui = mu;
  }

  Pk = cosmo->Plin(ki);
  Pk_nw = P_nw(ki);
  Pk_w = P_w(ki);

  Sigma2_tot = (1.0 + f * sqr(mui) * (2.0 + f)) * Sigma2 +
               sqr(f) * sqr(mui) * (sqr(mui) - 1.0) * deltaSigma2;

  Pk_lin_nw = sqr(1.0 + beta * sqr(mui)) * sqr(b1) * Pk_nw;
  Pk_lin_w = sqr(1.0 + beta * sqr(mui)) * sqr(b1) * Pk_w;

  if (ki < kmin || ki > kmax) {
    Pk_1l_nw = 0.0;
    Pk_1l_w = 0.0;
  } else {
    Pk_1l_nw = 0.0;
    Pk_1l_w = 0.0;
    for (int n = 0; n < 5; ++n) {
      Pk_1l_nw += pow(mui, 2.0 * n) *
                  gsl_spline_eval(spl_Pk_1l_nw[n], ki, acc_Pk_1l_nw[n]);
      Pk_1l_w += pow(mui, 2.0 * n) *
                 gsl_spline_eval(spl_Pk_1l_w[n], ki, acc_Pk_1l_w[n]);
    }
  }

  Pnw_ctr = -2.0 * c0 * sqr(ki) * Pk_lin_nw -
            2.0 * c2 * f * sqr(mui) * sqr(ki) * Pk_lin_nw -
            2.0 * c4 * sqr(f) * qua(mui) * sqr(ki) * Pk_lin_nw +
            cd4 * qua(f) * qua(mui) * qua(ki) * sqr(1.0 + beta * sqr(mui)) *
                sqr(b1) * Pk_lin_nw;
  Pw_ctr = -2.0 * c0 * sqr(ki) * Pk_lin_w -
           2.0 * c2 * f * sqr(mui) * sqr(ki) * Pk_lin_w -
           2.0 * c4 * sqr(f) * qua(mui) * sqr(ki) * Pk_lin_w +
           cd4 * qua(f) * qua(mui) * qua(ki) * sqr(1.0 + beta * sqr(mui)) *
               sqr(b1) * Pk_lin_w;

  Pkmui = Pk_lin_nw + Pk_1l_nw + Pnw_ctr +
          exp(-sqr(ki) * Sigma2_tot) *
              (Pk_lin_w * (1.0 + sqr(ki) * Sigma2_tot) + Pk_1l_w + Pw_ctr) +
          Pshot;

  if (flag_AP) {
    Pkmui /= cub(rs_drag_ratio) * sqr(alpha_perp) * alpha_para;
  }

  return Pkmui;
}

vector<vector<double>> IR_EFT::get_multipoles(vector<double> k, vector<int> l) {
  int nk, nl, li;
  double mui, wmui;
  vector<double> mu(nmuint), wmu(nmuint);
  vector<vector<double>> Pkmu;

  nk = k.size();
  nl = l.size();

  vector<vector<double>> res(nk, vector<double>(nl));

  for (int imu = 0; imu < nmuint; ++imu) {
    gsl_integration_glfixed_point(0.0, 1.0, imu, &mui, &wmui, t_mu);
    mu[imu] = mui;
    wmu[imu] = wmui;
  }

  Pkmu = calc_Pkmu_1l(k, mu);

  for (int il = 0; il < nl; ++il) {
    li = l[il];
    for (int ik = 0; ik < nk; ++ik) {
      res[ik][il] = 0.0;
      for (int imu = 0; imu < nmuint; ++imu) {
        res[ik][il] += wmu[imu] * (2.0 * li + 1.0) * Pkmu[ik][imu] *
                       gsl_sf_legendre_Pl(li, mu[imu]);
      }
    }
  }

  return res;
}

vector<vector<double>> IR_EFT::get_wedges(vector<double> k, vector<double> wedges) {
  int nk, nw;
  double mui, wmui;
  vector<double> mu(nmuint), wmu(nmuint);
  vector<vector<double>> Pkmu;

  nk = k.size();
  nw = wedges.size() - 1;

  vector<vector<double>> res(nk, vector<double>(nw));

  for (int iw = 0; iw < nw; ++iw) {
    for (int imu = 0; imu < nmuint; ++imu) {
      gsl_integration_glfixed_point(wedges[iw], wedges[iw + 1], imu, &mui,
                                    &wmui, t_mu);
      mu[imu] = mui;
      wmu[imu] = wmui;
    }

    Pkmu = calc_Pkmu_1l(k, mu);

    for (int ik = 0; ik < nk; ++ik) {
      res[ik][iw] = 0.0;
      for (int imu = 0; imu < nmuint; ++imu) {
        res[ik][iw] += wmu[imu] * Pkmu[ik][imu];
      }
      res[ik][iw] = res[ik][iw] / (wedges[iw + 1] - wedges[iw]);
    }
  }

  return res;
}



pair<vector<double>, vector<vector<double>>>
IR_EFT::get_multipoles_grid(vector<double> kbin, vector<int> l) {
  int nkbin, nl, ind, li;
  double ki, mui, Pkmui, kf, I1, I2, I3;
  vector<double> Nk, kmean;

  nl = l.size();
  nkbin = kbin.size() - 1;

  vector<vector<double>> res(nkbin, vector<double>(nl));
  Nk.resize(nkbin);
  kmean.resize(nkbin);

  kf = 2.0 * pi / L;

  for (int ikbin = 0; ikbin < nkbin; ++ikbin) {
    Nk[ikbin] = 0.0;
    kmean[ikbin] = 0.0;
    for (int il = 0; il < nl; ++il)
      res[ikbin][il] = 0.0;
  }

  for (int i1 = 0; i1 < ng; ++i1) {
    I1 = (i1 < ng / 2) ? (i1) : (ng - i1);
    for (int i2 = 0; i2 < ng; ++i2) {
      I2 = (i2 < ng / 2) ? (i2) : (ng - i2);
      for (int i3 = 0; i3 < ng; ++i3) {
        I3 = (i3 < ng / 2) ? (i3) : (ng - i3);

        if (i1 == 0 && i2 == 0 && i3 == 0)
          continue;

        ki = kf * sqrt(sqr(I1) + sqr(I2) + sqr(I3));
        mui = I3 * kf / ki;

        ind = -1;
        for (int ikbin = 0; ikbin < nkbin; ++ikbin) {
          if (kbin[ikbin] <= ki && ki < kbin[ikbin + 1]) {
            ind = ikbin;
          }
        }
        if (ind == -1)
          continue;

        Pkmui = calc_Pkmu_1l(ki, mui);
        Nk[ind] += 1.0;
        kmean[ind] += ki;

        for (int il = 0; il < nl; ++il) {
          li = l[il];
          res[ind][il] +=
              (2.0 * li + 1.0) * Pkmui * gsl_sf_legendre_Pl(li, mui);
        }
      }
    }
  }

  for (int ikbin = 0; ikbin < nkbin; ++ikbin) {
    if (Nk[ikbin] == 0)
      continue;
    kmean[ikbin] /= Nk[ikbin];
    for (int il = 0; il < nl; ++il)
      res[ikbin][il] /= Nk[ikbin];
  }

  return make_pair(kmean, res);
}


tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>>
IR_EFT::get_wedges_grid(vector<double> kbin, vector<double> wedges) {
  int nkbin, nw, ind;
  double ki, mui, Pkmui, kf, kmin, kmax, I1, I2, I3;

  nw = wedges.size() - 1;
  nkbin = kbin.size() - 1;

  vector<vector<double>> res(nkbin, vector<double>(nw));
  vector<vector<double>> Nkmu(nkbin, vector<double>(nw));
  vector<vector<double>> kmean(nkbin, vector<double>(nw));
  vector<vector<double>> mumean(nkbin, vector<double>(nw));

  kmin = kbin[0];
  kmax = kbin[nkbin];
  kf = 2.0 * pi / L;

  for (int ikbin = 0; ikbin < nkbin; ++ikbin) {
    for (int iw = 0; iw < nw; ++iw) {
      res[ikbin][iw] = 0.0;
      Nkmu[ikbin][iw] = 0.0;
      kmean[ikbin][iw] = 0.0;
      mumean[ikbin][iw] = 0.0;
    }
  }

  for (int i1 = 0; i1 < ng; ++i1) {
    I1 = (i1 < ng / 2) ? (i1) : (ng - i1);
    for (int i2 = 0; i2 < ng; ++i2) {
      I2 = (i2 < ng / 2) ? (i2) : (ng - i2);
      for (int i3 = 0; i3 < ng; ++i3) {
        I3 = (i3 < ng / 2) ? (i3) : (ng - i3);

        if (i1 == 0 && i2 == 0 && i3 == 0)
          continue;
        ki = kf * sqrt(sqr(I1) + sqr(I2) + sqr(I3));
        mui = I3 * kf / ki;

        ind = -1;
        for (int ikbin = 0; ikbin < nkbin; ++ikbin) {
          if (kbin[ikbin] <= ki && ki < kbin[ikbin + 1]) {
            ind = ikbin;
          }
        }
        if (ind == -1)
          continue;

        Pkmui = calc_Pkmu_1l(ki, mui);

        if (I3 == 0) {
          // mu = 0
          Nkmu[ind][0] += 1.0;
          res[ind][0] += Pkmui;
          kmean[ind][0] += ki;
          mumean[ind][0] += mui;
        } else if (I1 == 0 && I2 == 0) {
          // mu = 1
          Nkmu[ind][nw - 1] += 1.0;
          res[ind][nw - 1] += Pkmui;
          kmean[ind][nw - 1] += ki;
          mumean[ind][nw - 1] += mui;
        } else {
          for (int iw = 0; iw < nw; ++iw) {
            if (wedges[iw] <= mui && mui < wedges[iw + 1]) {
              Nkmu[ind][iw] += 1.0;
              res[ind][iw] += Pkmui;
              kmean[ind][iw] += ki;
              mumean[ind][iw] += mui;
            }
          }
        }
      }
    }
  }

  for (int ikbin = 0; ikbin < nkbin; ++ikbin) {
    for (int iw = 0; iw < nw; ++iw) {
      if (Nkmu[ikbin][iw] == 0)
        continue;
      res[ikbin][iw] /= Nkmu[ikbin][iw];
      kmean[ikbin][iw] /= Nkmu[ikbin][iw];
      mumean[ikbin][iw] /= Nkmu[ikbin][iw];
    }
  }

  return make_tuple(kmean, mumean, res);
}


/*
double IR_EFT::Z1(Vector k1){
  return b1+f*sqr(k1.z/sqrt(k1*k1));
}

double IR_EFT::Z2(Vector k1, Vector k2){
  double kk1, kk2, kk, mu1, mu2, mu, res;

  kk1 = sqrt(k1*k1);
  kk2 = sqrt(k2*k2);
  kk = sqrt((k1+k2)*(k1+k2));
  mu1 = k1.z/kk1;
  mu2 = k2.z/kk2;
  mu = (k1.z+k2.z)/kk;

  res = 0.5*b2 + bG2*((k1*k2)/((k1*k1)*(k2*k2)) - 1.0) + b1*F2_sym(DENS, k1, k2)
+ f*sqr(mu)*F2_sym(VELO, k1, k2) + 0.5*f*mu*kk* (mu1/kk1*(b1+f*sqr(mu1)) +
mu2/kk2*(b1+f*sqr(mu2)));

  return res;
}

double IR_EFT::Z3(Vector k1, Vector k2, Vector k3){
  return (Z3_unsym(k1, k2, k3)+Z3_unsym(k1, k3, k2)
         +Z3_unsym(k2, k1, k3)+Z3_unsym(k2, k3, k1)
         +Z3_unsym(k3, k1, k2)+Z3_unsym(k3, k2, k1))/6.0;
}

double IR_EFT::Z3_unsym(Vector k1, Vector k2, Vector k3){
  double res;
  double kk, kk1, kk2, kk3, kk12, kk23;
  double mu, mu1, mu2, mu3, mu12, mu23;


  kk1 = sqrt(k1*k1);
  kk2 = sqrt(k2*k2);
  kk2 = sqrt(k3*k3);
  kk12 = sqrt((k1+k2)*(k1+k2));
  kk23 = sqrt((k2+k3)*(k2+k3));
  kk = sqrt((k1+k2+k3)*(k1+k2+k3));
  mu1 = k1.z/kk1;
  mu2 = k2.z/kk2;
  mu3 = k3.z/kk3;
  mu12 = (k1.z+k2.z)/kk12;
  mu23 = (k2.z+k3.z)/kk23;
  mu = (k1.z+k2.z+k3.z)/kk;


  res += 2.0*bGamma2*(sqr(k1*(k2+k3))/(sqr(kk1)*sqr(kk23)) - 1.0)*(F2_sym(DENS,
k2, k3)-F2_sym(VELO, k2, k3)) + b1*F3_sym(DENS, k1, k2, k3) +
f*sqr(mu)*F3_sym(VELO, k1, k2, k3) +
         0.5*sqr(f*mu*kk)*(b1+f*sqr(mu1))*mu2/kk2*mu3/kk3 +
         f*mu*kk*mu3/kk3*(b1*F2_sym(DENS, k1, k2)+f*sqr(mu12)*F2_sym(VELO, k1,
k2)) + f*mu*kk*(b1+f*sqr(mu1))*mu23/kk23*F2_sym(VELO, k2, k3) + b2*F2_sym(DENS,
k1, k2) + 2.0*bG2*(sqr(k1*(k2+k3))/(sqr(kk1)*sqr(kk23)) - 1.0)*F2_sym(DENS, k2,
k3) + 0.5*b2*f*mu*kk*mu1/kk1 +
bG2*f*mu*kk*mu1/kk1*(sqr(k2*k3)/(sqr(kk2)*sqr(kk3)) - 1.0);

  return res;
}


double IR_EFT::K2(double k, double q, double x){
  double res1, res2, res;


  res1 =
b1/(2.0*q)*(-2.0*cub(q)+cub(k)*x+4.0*k*sqr(q)*x-sqr(k)*q*(1.0+2.0*sqr(x)))/(sqr(k)+sqr(q)-2.0*k*q*x);
  res2 =
b2/7.0*(7.0*sqr(q)-14.0*k*q*x+sqr(k)*(5.0+2.0*sqr(x)))/(sqr(k)+sqr(q)-2.0*k*q*x);

  res = res1 + res2 + b4;

  return res;
}

double IR_EFT::K3(double k, double q){
  double res1, res2, res;
  double logkq;


  if(abs((k-q)/(k+q)) < 1e-10) logkq = 0.0;
  else logkq = log(abs((k-q)/(k+q)));

  res1 =
b1/(504.0*cub(k*q))*(-38.0*cub(k)*sqr(k)*q+48.0*cub(k*q)-18.0*k*cub(q)*sqr(q)+9.0*cub(sqr(k)-sqr(q))*logkq);
  res2 =
b3/(756.0*cub(k*q)*sqr(q))*(2.0*k*q*(sqr(k)+sqr(q))*(3.0*qua(k)-14.0*sqr(k*q)+3.0*qua(q))+3.0*qua(sqr(k)-sqr(q))*logkq);

  res = res1 + res2;

  return res;
}
*/

double IR_EFT::A_func(int n, int m, double r, double x) {
  if (n == 0 && m == 0) {
    return 1.0 / 98.0 * sqr(3.0 * r + 7.0 * x - 10.0 * r * sqr(x));
  } else if (n == 1 && m == 1) {
    return 2.0 / 49.0 * sqr(3.0 * r + 7.0 * x - 10.0 * r * sqr(x));
  } else if (n == 1 && m == 2) {
    return 1.0 / 28.0 * (1.0 - sqr(x)) *
           (7.0 - 6.0 * sqr(r) - 42.0 * r * x + 48.0 * sqr(r * x));
  } else if (n == 2 && m == 2) {
    return 1.0 / 196.0 *
           (-49.0 + 637.0 * sqr(x) + 42.0 * r * x * (17.0 - 45.0 * sqr(x)) +
            6.0 * sqr(r) * (19.0 - 157.0 * sqr(x) + 236.0 * qua(x)));
  } else if (n == 2 && m == 3) {
    return 1.0 / 14.0 * (1.0 - sqr(x)) *
           (7.0 - 42.0 * r * x - 6.0 * sqr(r) + 48.0 * sqr(r * x));
  } else if (n == 2 && m == 4) {
    return 3.0 / 16.0 * sqr(r) * sqr(1.0 - sqr(x));
  } else if (n == 3 && m == 3) {
    return 1.0 / 14.0 *
           (-7.0 + 35.0 * sqr(x) + 54.0 * r * x - 110.0 * r * cub(x) +
            6.0 * sqr(r) - 66.0 * sqr(r * x) + 88.0 * sqr(r * sqr(x)));
  } else if (n == 3 && m == 4) {
    return 1.0 / 8.0 * (1.0 - sqr(x)) *
           (2.0 - 3.0 * sqr(r) - 12.0 * r * x + 15.0 * sqr(r * x));
  } else if (n == 4 && m == 4) {
    return 1.0 / 16.0 *
           (-4.0 + 12.0 * sqr(x) + 3.0 * sqr(r) + 24.0 * r * x -
            30.0 * sqr(r * x) - 40.0 * r * cub(x) + 35.0 * sqr(r * sqr(x)));
  } else {
    return 0.0;
  }
}

double IR_EFT::B_func(int n, int m, double r) {
  double logr;

  if (fabs(1.0 - r) < 1e-10) {
    logr = 0.0;
  } else {
    logr = log(fabs((1.0 + r) / (1.0 - r)));
  }

  if (n == 0 && m == 0) {
    return 1.0 / 252.0 *
           (12.0 / sqr(r) - 158.0 + 100.0 * sqr(r) - 42.0 * qua(r) +
            3.0 / cub(r) * cub(sqr(r) - 1.0) * (7.0 * sqr(r) + 2.0) * logr);
  } else if (n == 1 && m == 1) {
    return 1.0 / 84.0 *
           (12.0 / sqr(r) - 158.0 + 100.0 * sqr(r) - 42.0 * qua(r) +
            3.0 / cub(r) * cub(sqr(r) - 1.0) * (7.0 * sqr(r) + 2.0) * logr);
  } else if (n == 1 && m == 2) {
    return 1.0 / 168.0 *
           (18.0 / sqr(r) - 178.0 - 66.0 * sqr(r) + 18.0 * qua(r) -
            9.0 / cub(r) * qua(sqr(r) - 1.0) * logr);
  } else if (n == 2 && m == 2) {
    return 1.0 / 168.0 *
           (18.0 / sqr(r) - 218.0 + 126.0 * sqr(r) - 54.0 * qua(r) +
            9.0 / cub(r) * cub(sqr(r) - 1.0) * (3.0 * sqr(r) + 1.0) * logr);
  } else if (n == 2 && m == 3) {
    return -2.0 / 3.0;
  } else {
    return 0.0;
  }
}
