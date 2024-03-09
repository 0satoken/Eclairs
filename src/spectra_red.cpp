#include "spectra_red.hpp"
#include "cosmology.hpp"
#include "fast_spectra.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <string>
#include <utility>


spectra_red::spectra_red(params &para, cosmology &cosmo, direct_red &dir_red) {
  this->dir_red = &dir_red;
  this->cosmo = &cosmo;
  flag_fast = false;
  flag_direct_spline = para.bparams["direct_spline"];
  flag_SPT = para.bparams["direct_SPT"];
  flag_sigma_vlin = para.bparams["use_sigma_vlin"];

  pi = 4.0 * atan(1.0);
  eta = cosmo.get_eta();       // exponential growth factor
  f = cosmo.get_growth_rate(); // growth rate
  
  if(flag_SPT || flag_sigma_vlin){
    sigma_vlin = exp(eta) * cosmo.get_linear_displacement_dispersion();
  }

  b1 = para.dparams["b1"];
  beta = f / b1;
  if(flag_sigma_vlin){
    sigma_v = sigma_vlin;
  }
  else{
    sigma_v = para.dparams["sigma_v"];
  }

  alpha_perp = para.dparams["alpha_perp"];
  alpha_para = para.dparams["alpha_para"];
  rs_drag_ratio = para.dparams["rs_drag_ratio"];

  flag_AP = para.bparams["AP"];
  FoG_type = para.sparams["FoG_type"];
  gamma = para.dparams["gamma"];

  nmu = para.iparams["multipole_nmu"];
  L = para.dparams["grid_L"];
  ng = para.iparams["grid_ng"];

  t_mu = gsl_integration_glfixed_table_alloc(nmu);
}

spectra_red::spectra_red(params &para, cosmology &cosmo, fast_spectra &fast_spec) {
  this->fast_spec = &fast_spec;
  this->cosmo = &cosmo;
  flag_fast = true;
  flag_direct_spline = false;
  flag_sigma_vlin = para.bparams["use_sigma_vlin"];

  pi = 4.0 * atan(1.0);
  eta = cosmo.get_eta();       // exponential growth factor
  f = cosmo.get_growth_rate(); // growth rate

  b1 = para.dparams["b1"];
  beta = f / b1;
  if(flag_sigma_vlin){
    sigma_vlin = exp(eta) * cosmo.get_linear_displacement_dispersion();
    sigma_v = sigma_vlin;
  }
  else{
    sigma_v = para.dparams["sigma_v"];
  }
  alpha_perp = para.dparams["alpha_perp"];
  alpha_para = para.dparams["alpha_para"];
  rs_drag_ratio = para.dparams["rs_drag_ratio"];

  flag_AP = para.bparams["AP"];
  FoG_type = para.sparams["FoG_type"];
  gamma = para.dparams["gamma"];

  nmu = para.iparams["multipole_nmu"];
  L = para.dparams["grid_L"];
  ng = para.iparams["grid_ng"];

  t_mu = gsl_integration_glfixed_table_alloc(nmu);
}

spectra_red::~spectra_red() {
  gsl_integration_glfixed_table_free(t_mu);
}

double spectra_red::D_FoG(double x) {
  double res;

  if (FoG_type == "Gaussian" || FoG_type == "gaussian") {
    res = exp(-x * x);
  } else if (FoG_type == "Lorentzian" || FoG_type == "lorentzian") {
    res = 1.0 / sqr(1.0 + 0.5 * sqr(x));
  } else if (FoG_type == "Gamma" || FoG_type == "gamma") {
    res = pow(1.0 + sqr(x) / gamma, -gamma);
  } else if (FoG_type == "None" || FoG_type == "none") {
    res = 1.0;
  } else {
    cerr << "[ERROR] Invalid FoG type:" << FoG_type << endl;
    exit(1);
  }

  return res;
}

double spectra_red::get_2D_power(double k, double mu) {
  double ki, muj, qi, nuj, D0, Pdd, Pdt, Ptt, Plin, PKaiser, P1loop;
  double A2, A4, A6, B2, B4, B6, B8, C2, C4, C6, C8;
  double res;
  map<string, double> Pk, Aterm, Bterm, Cterm;

  ki = k;
  muj = mu;

  if (flag_AP) {
    qi = ki / alpha_perp *
         sqrt(1.0 + sqr(muj) * (sqr(alpha_perp / alpha_para) - 1.0));
    nuj = muj * alpha_perp / alpha_para /
          sqrt(1.0 + sqr(muj) * (sqr(alpha_perp / alpha_para) - 1.0));
  } else {
    qi = ki;
    nuj = muj;
  }

  if (flag_fast) {
    Pk = fast_spec->get_spectra_2l(qi);
    Aterm = fast_spec->get_Aterm(qi);
    Bterm = fast_spec->get_Bterm(qi);
  } else {
    if (flag_direct_spline) {
      Pk = dir_red->get_spl_spectra(qi);
      Aterm = dir_red->get_spl_Aterm(qi);
      Bterm = dir_red->get_spl_Bterm(qi);
      if (flag_SPT)
        Cterm = dir_red->get_spl_Cterm(qi);
    } else {
      Pk = dir_red->get_spectra(qi);
      Aterm = dir_red->get_Aterm(qi);
      Bterm = dir_red->get_Bterm(qi);
      if (flag_SPT)
        Cterm = dir_red->get_Cterm(qi);
    }
  }

  Plin = cosmo->Plin(qi);
  Pdd = Pk["dd"];
  Pdt = Pk["dt"];
  Ptt = Pk["tt"];
  A2 = Aterm["A2"];
  A4 = Aterm["A4"];
  A6 = Aterm["A6"];
  B2 = Bterm["B2"];
  B4 = Bterm["B4"];
  B6 = Bterm["B6"];
  B8 = Bterm["B8"];

  if (!flag_fast && flag_SPT) {
    C2 = Cterm["C2"];
    C4 = Cterm["C4"];
    C6 = Cterm["C6"];
    C8 = Cterm["C8"];
  }

  if (!flag_fast && flag_SPT) {
    D0 = 1.0 - sqr(qi * nuj * f * sigma_vlin);
    PKaiser = sqr(b1) * sqr(1.0 + beta * sqr(nuj)) * Plin;
    P1loop = sqr(b1) * (Pdd + 2.0 * beta * sqr(nuj) * Pdt +
                        sqr(beta) * qua(nuj) * Ptt) -
             PKaiser;

    res = D0 * PKaiser + P1loop +
          cub(b1) * (sqr(nuj) * A2 + qua(nuj) * A4 + sqr(nuj) * qua(nuj) * A6) +
          qua(b1) * (sqr(nuj) * B2 + qua(nuj) * B4 + sqr(nuj) * qua(nuj) * B6 +
                     qua(nuj) * qua(nuj) * B8) +
          qua(b1) * (sqr(nuj) * C2 + qua(nuj) * C4 + sqr(nuj) * qua(nuj) * C6 +
                     qua(nuj) * qua(nuj) * C8);
  } else {
    D0 = D_FoG(qi * nuj * f * sigma_v);
    res = D0 * sqr(b1) *
          (Pdd + 2.0 * beta * sqr(nuj) * Pdt + sqr(beta) * qua(nuj) * Ptt +
           b1 * (sqr(nuj) * A2 + qua(nuj) * A4 + sqr(nuj) * qua(nuj) * A6) +
           sqr(b1) * (sqr(nuj) * B2 + qua(nuj) * B4 + sqr(nuj) * qua(nuj) * B6 +
                      qua(nuj) * qua(nuj) * B8));
  }

  if (flag_AP) {
    res /= cub(rs_drag_ratio) * sqr(alpha_perp) * alpha_para;
  }

  return res;
}

vector<vector<double>> spectra_red::get_2D_power(vector<double> k, vector<double> mu) {
  int nk, nmu;
  double ki, muj, qi, nuj, D0, Pdd, Pdt, Ptt, Plin;
  double A2, A4, A6, B2, B4, B6, B8, C2, C4, C6, C8;
  map<string, double> Pk, Aterm, Bterm, Cterm, bias_terms;

  nk = k.size();
  nmu = mu.size();
  vector<vector<double>> res(nk, vector<double>(nmu));

  if (flag_AP) {
    for (int i = 0; i < nk; ++i) {
      ki = k[i];
      for (int j = 0; j < nmu; ++j) {
        muj = mu[j];

        qi = ki / alpha_perp *
             sqrt(1.0 + sqr(muj) * (sqr(alpha_perp / alpha_para) - 1.0));
        nuj = muj * alpha_perp / alpha_para /
              sqrt(1.0 + sqr(muj) * (sqr(alpha_perp / alpha_para) - 1.0));

        if (flag_fast) {
          Pk = fast_spec->get_spectra_2l(qi);
          Aterm = fast_spec->get_Aterm(qi);
          Bterm = fast_spec->get_Bterm(qi);
        } else {
          if (flag_direct_spline) {
            Pk = dir_red->get_spl_spectra(qi);
            Aterm = dir_red->get_spl_Aterm(qi);
            Bterm = dir_red->get_spl_Bterm(qi);
            if (flag_SPT)
              Cterm = dir_red->get_spl_Cterm(qi);
          } else {
            Pk = dir_red->get_spectra(qi);
            Aterm = dir_red->get_Aterm(qi);
            Bterm = dir_red->get_Bterm(qi);
            if (flag_SPT)
              Cterm = dir_red->get_Cterm(qi);
          }
        }

        Pdd = Pk["dd"];
        Pdt = Pk["dt"];
        Ptt = Pk["tt"];
        A2 = Aterm["A2"];
        A4 = Aterm["A4"];
        A6 = Aterm["A6"];
        B2 = Bterm["B2"];
        B4 = Bterm["B4"];
        B6 = Bterm["B6"];
        B8 = Bterm["B8"];

        if (!flag_fast && flag_SPT) {
          C2 = Cterm["C2"];
          C4 = Cterm["C4"];
          C6 = Cterm["C6"];
          C8 = Cterm["C8"];
        }

        D0 = D_FoG(qi * nuj * f * sigma_v);
        res[i][j] =
            D0 * sqr(b1) *
            (Pdd + 2.0 * beta * sqr(nuj) * Pdt + sqr(beta) * qua(nuj) * Ptt +
             b1 * (sqr(nuj) * A2 + qua(nuj) * A4 + sqr(nuj) * qua(nuj) * A6) +
             sqr(b1) * (sqr(nuj) * B2 + qua(nuj) * B4 +
                        sqr(nuj) * qua(nuj) * B6 + qua(nuj) * qua(nuj) * B8));

        if (!flag_fast && flag_SPT) {
          res[i][j] += D0 * sqr(b1) * sqr(b1) *
                       (sqr(nuj) * C2 + qua(nuj) * C4 +
                        sqr(nuj) * qua(nuj) * C6 + qua(nuj) * qua(nuj) * C8);
        }

        res[i][j] /= cub(rs_drag_ratio) * sqr(alpha_perp) * alpha_para;
      }
    }
  } else {
    for (int i = 0; i < nk; ++i) {
      ki = k[i];
      if (flag_fast) {
        Pk = fast_spec->get_spectra_2l(ki);
        Aterm = fast_spec->get_Aterm(ki);
        Bterm = fast_spec->get_Bterm(ki);
      } else {
        if (flag_direct_spline) {
          Pk = dir_red->get_spl_spectra(ki);
          Aterm = dir_red->get_spl_Aterm(ki);
          Bterm = dir_red->get_spl_Bterm(ki);
          if (flag_SPT)
            Cterm = dir_red->get_spl_Cterm(ki);
        } else {
          Pk = dir_red->get_spectra(ki);
          Aterm = dir_red->get_Aterm(ki);
          Bterm = dir_red->get_Bterm(ki);
          if (flag_SPT)
            Cterm = dir_red->get_Cterm(ki);
        }
      }

      Pdd = Pk["dd"];
      Pdt = Pk["dt"];
      Ptt = Pk["tt"];
      A2 = Aterm["A2"];
      A4 = Aterm["A4"];
      A6 = Aterm["A6"];
      B2 = Bterm["B2"];
      B4 = Bterm["B4"];
      B6 = Bterm["B6"];
      B8 = Bterm["B8"];

      if (!flag_fast && flag_SPT) {
        C2 = Cterm["C2"];
        C4 = Cterm["C4"];
        C6 = Cterm["C6"];
        C8 = Cterm["C8"];
      }

      for (int j = 0; j < nmu; ++j) {
        muj = mu[j];

        D0 = D_FoG(ki * muj * f * sigma_v);
        res[i][j] =
            D0 * sqr(b1) *
            (Pdd + 2.0 * beta * sqr(muj) * Pdt + sqr(beta) * qua(muj) * Ptt +
             b1 * (sqr(muj) * A2 + qua(muj) * A4 + sqr(muj) * qua(muj) * A6) +
             sqr(b1) * (sqr(muj) * B2 + qua(muj) * B4 +
                        sqr(muj) * qua(muj) * B6 + qua(muj) * qua(muj) * B8));

        if (!flag_fast && flag_SPT) {
          res[i][j] += D0 * sqr(b1) * sqr(b1) *
                       (sqr(muj) * C2 + qua(muj) * C4 +
                        sqr(muj) * qua(muj) * C6 + qua(muj) * qua(muj) * C8);
        }
      }
    }
  }

  return res;
}

vector<vector<double>> spectra_red::get_multipoles(vector<double> k,
                                                   vector<int> l) {
  int nk, nl, li;
  double mui, wmui;
  vector<double> mu(nmu), wmu(nmu);
  vector<vector<double>> Pkmu;

  nk = k.size();
  nl = l.size();

  vector<vector<double>> res(nk, vector<double>(nl));

  for (int imu = 0; imu < nmu; ++imu) {
    gsl_integration_glfixed_point(0.0, 1.0, imu, &mui, &wmui, t_mu);
    mu[imu] = mui;
    wmu[imu] = wmui;
  }

  Pkmu = get_2D_power(k, mu);

  for (int il = 0; il < nl; ++il) {
    li = l[il];
    for (int ik = 0; ik < nk; ++ik) {
      res[ik][il] = 0.0;
      for (int imu = 0; imu < nmu; ++imu) {
        res[ik][il] += wmu[imu] * (2.0 * li + 1.0) * Pkmu[ik][imu] *
                       gsl_sf_legendre_Pl(li, mu[imu]);
      }
    }
  }

  return res;
}

vector<vector<double>> spectra_red::get_wedges(vector<double> k,
                                               vector<double> wedges) {
  int nk, nw;
  double mui, wmui;
  vector<double> mu(nmu), wmu(nmu);
  vector<vector<double>> Pkmu;

  nk = k.size();
  nw = wedges.size() - 1;

  vector<vector<double>> res(nk, vector<double>(nw));

  for (int iw = 0; iw < nw; ++iw) {
    for (int imu = 0; imu < nmu; ++imu) {
      gsl_integration_glfixed_point(wedges[iw], wedges[iw + 1], imu, &mui,
                                    &wmui, t_mu);
      mu[imu] = mui;
      wmu[imu] = wmui;
    }

    Pkmu = get_2D_power(k, mu);

    for (int ik = 0; ik < nk; ++ik) {
      res[ik][iw] = 0.0;
      for (int imu = 0; imu < nmu; ++imu) {
        res[ik][iw] += wmu[imu] * Pkmu[ik][imu];
      }
      res[ik][iw] = res[ik][iw] / (wedges[iw + 1] - wedges[iw]);
    }
  }

  return res;
}

pair<vector<double>, vector<vector<double>>>
spectra_red::get_multipoles_grid(vector<double> kbin, vector<int> l) {
  int nkbin, nl, ind, li;
  double ki, mui, Pkmui, kf, kmin, kmax, I1, I2, I3;
  vector<double> Nk, kmean;

  nl = l.size();
  nkbin = kbin.size() - 1;

  vector<vector<double>> res(nkbin, vector<double>(nl));
  Nk.resize(nkbin);
  kmean.resize(nkbin);

  kf = 2.0 * pi / L;
  kmin = kbin[0];
  kmax = kbin[nkbin];

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

        Pkmui = get_2D_power(ki, mui);
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
spectra_red::get_wedges_grid(vector<double> kbin, vector<double> wedges) {
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

        Pkmui = get_2D_power(ki, mui);

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

double spectra_red::conversion_multipole(int l, int n, double alpha) {
  double res;

  if (l == 0) {
    res = 0.5 * pow(alpha, -n - 0.5) * gamma_func(n + 0.5, alpha);
  } else if (l == 2) {
    res =
        -5.0 / 4.0 * pow(alpha, -n - 1.5) *
        (alpha * gamma_func(n + 0.5, alpha) - 3.0 * gamma_func(n + 1.5, alpha));
  } else if (l == 4) {
    res = -9.0 / 64.0 * pow(alpha, -n - 1.5) *
          (12.0 * sqr(alpha) * gamma_func(n + 0.5, alpha) -
           120.0 * alpha * gamma_func(n + 1.5, alpha) +
           140.0 * gamma_func(n + 2.5, alpha));
  } else {
    cerr << "[ERROR] Invalid l: " << l << endl;
    cerr << "l should be either 0 (monopole), 2 (quadrupole), 4 (hexadecapole)."
         << endl;
    exit(1);
  }

  return res;
}

double spectra_red::conversion_wedge(double mu1, double mu2, int n,
                                     double alpha) {
  double res;

  if (mu1 > mu2) {
    cerr << "[ERROR] mu1 should be smaller than mu2." << endl;
    exit(1);
  }

  res = -0.5 * pow(mu2, 2.0 * n + 1.0) * pow(alpha * sqr(mu2), -n - 0.5) * gsl_sf_gamma_inc(n + 0.5, alpha * mu2) +
        0.5 * pow(mu1, 2.0 * n + 1.0) * pow(alpha * sqr(mu1), -n - 0.5) * gsl_sf_gamma_inc(n + 0.5, alpha * mu1);
  res = res / (mu2 - mu1);

  return res;
}

double spectra_red::gamma_func(double a, double x) {
  return gsl_sf_gamma_inc_P(a, x) * gsl_sf_gamma(a);
}
