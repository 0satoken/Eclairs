#include "spectra.hpp"
#include "cosmology.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_spline.h>
#include <iostream>
#include <string>


spectra::spectra(params &para, cosmology &cosmo) {
  this->cosmo = &cosmo;

  pi = 4.0 * atan(1.0);
  eta = cosmo.get_eta();
  kmin = para.dparams["kmin"];
  kmax = para.dparams["kmax"];
  lambda = para.dparams["lambda_power"];

  flag_verbose = para.bparams["verbose"];

  /* setup for numerical integration */
  nint = para.iparams["nint"];
  nk_G1_1 = para.iparams["nk_G1_1"];
  nk_G1_2 = para.iparams["nk_G1_2"];
  nk_G2_1 = para.iparams["nk_G2_1"];
  nx = para.iparams["nx_Pcorr2"];
  nmu = para.iparams["nmu_Pcorr2"];

  t = gsl_integration_glfixed_table_alloc(nint);
  t_G1_1 = gsl_integration_glfixed_table_alloc(nk_G1_1);
  t_G1_2 = gsl_integration_glfixed_table_alloc(nk_G1_2);
  t_G2_1 = gsl_integration_glfixed_table_alloc(nk_G2_1);
  t_x = gsl_integration_glfixed_table_alloc(nx);
  t_mu = gsl_integration_glfixed_table_alloc(nmu);

  /* Monte-Carlo integration configuration */
  dim = 5;
  MC_calls = para.iparams["MC_calls"];
  MC_tol = para.dparams["MC_tol"];
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  s = gsl_monte_vegas_alloc(dim);

  /* RegPT+ setting */
  if (para.bparams["free_sigma_d"]) {
    flag_free_sigma_d = true;
    sigma_d = para.dparams["sigma_d"];
  } else {
    flag_free_sigma_d = false;
  }

  /* output setting */
  kminout = para.dparams["output_kmin"];
  kmaxout = para.dparams["output_kmax"];
  nkout = para.iparams["output_nk"];
  output_fname = para.sparams["output_file_name"];
  model = para.sparams["output_model"];
  spacing = para.sparams["output_spacing"];
  flag_1loop = para.bparams["output_1loop"];
  flag_output = para.bparams["output"];

  if (flag_output)
    output_spectra();
}

spectra::~spectra() {
  /* freeing memories */
  gsl_integration_glfixed_table_free(t);
  gsl_integration_glfixed_table_free(t_G1_1);
  gsl_integration_glfixed_table_free(t_G1_2);
  gsl_integration_glfixed_table_free(t_G2_1);
  gsl_integration_glfixed_table_free(t_x);
  gsl_integration_glfixed_table_free(t_mu);
  gsl_monte_vegas_free(s);
  gsl_rng_free(r);
}

/* return linear power spectrum at z=0 */
double spectra::P0(double k) { return cosmo->P0(k); }

/* return linear power spectrum at the redshift of cosmology class */
double spectra::Plin(double k) { return cosmo->Plin(k); }

/* return no-wiggle power spectrum at z=0 */
double spectra::Pno_wiggle0(double k) { return cosmo->Pno_wiggle0(k); }

/* return no-wiggle power spectrum at the redshift of the cosmology class */
double spectra::Pno_wiggle(double k) { return cosmo->Pno_wiggle(k); }

/*
 * this function gives the dispersion of velocity
 * with running UV cutoff k_lambda = k/2 as the fiducial model.
 * The user determined "sigma_d" is available.
 */
double spectra::get_sigmad2(double k, double lambda) {
  double ki, wi, res, k_lambda;

  if (flag_free_sigma_d) {
    return sqr(sigma_d);
  }

  k_lambda = k / lambda;

  if (k_lambda < kmin) {
    res = 0.0;
  } else if (k_lambda > kmax) {
    res = 0.0;
    for (int i = 0; i < nint; ++i) {
      gsl_integration_glfixed_point(log(kmin), log(kmax), i, &ki, &wi, t);
      ki = exp(ki);
      res += wi * ki * P0(ki);
    }
    res /= 6.0 * sqr(pi);
  } else {
    res = 0.0;
    for (int i = 0; i < nint; ++i) {
      gsl_integration_glfixed_point(log(kmin), log(k_lambda), i, &ki, &wi, t);
      ki = exp(ki);
      res += wi * ki * P0(ki);
    }
    res /= 6.0 * sqr(pi);
  }

  return res;
}

double spectra::Gamma1_1loop(Type a, double k) {
  double qi, wi, xi, xmin, xmax, res;

  res = 0.0;
  xmin = kmin / k;
  xmax = kmax / k;

  for (int i = 0; i < nk_G1_1; i++) {
    gsl_integration_glfixed_point(log(xmin), log(xmax), i, &xi, &wi, t_G1_1);
    qi = exp(xi) * k;
    res += wi * cub(qi) * kernel_Gamma1_1loop(a, k, qi) * P0(qi);
  }

  res /= (2.0 * pi * pi);

  return res;
}

double spectra::Gamma1_2loop(Type a, double k) {
  double res, integ;
  double *q, *w, *p0;

  q = new double[nk_G1_2];
  w = new double[nk_G1_2];
  p0 = new double[nk_G1_2];

  res = 0.0;

  for (int i = 0; i < nk_G1_2; i++) {
    gsl_integration_glfixed_point(log(kmin), log(kmax), i, &q[i], &w[i],
                                  t_G1_2);
    q[i] = exp(q[i]);
    p0[i] = P0(q[i]);
  }

  for (int i1 = 0; i1 < nk_G1_2; i1++) {
    integ = 0.0;
    for (int i2 = 0; i2 < nk_G1_2; i2++) {
      integ += w[i2] * kernel_Gamma1_2loop(a, k, q[i1], q[i2]) * p0[i2] * cub(q[i2]);
    }
    res += w[i1] * integ * p0[i1] * cub(q[i1]);
  }

  res /= sqr(2.0 * pi * pi);
  res += 0.5 * Gamma1_1loop(a, k) * Gamma1_1loop(a, k);

  delete[] q;
  delete[] w;
  delete[] p0;

  return res;
}

double spectra::Gamma2_tree(Type a, double k1, double k2, double k3) {
  double k1dk2, res;

  k1dk2 = 0.5 * (k3 * k3 - k1 * k1 - k2 * k2);

  switch (a) {
  case DENS:
    res = 5.0 / 7.0 + 0.5 * k1dk2 * (1.0 / (k1 * k1) + 1.0 / (k2 * k2)) +
          2.0 / 7.0 * sqr(k1dk2 / (k1 * k2));
    break;
  case VELO:
    res = 3.0 / 7.0 + 0.5 * k1dk2 * (1.0 / (k1 * k1) + 1.0 / (k2 * k2)) +
          4.0 / 7.0 * sqr(k1dk2 / (k1 * k2));
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}

double spectra::Gamma2_1loop(Type a, double k1, double k2, double k3) {
  double qi, wi, res;

  res = 0.0;

  for (int i = 0; i < nk_G2_1; i++) {
    gsl_integration_glfixed_point(log(kmin), log(kmax), i, &qi, &wi, t_G2_1);
    qi = exp(qi);
    res += wi * cub(qi) * kernel_Gamma2_1loop(a, k1, k2, k3, qi) * P0(qi);
  }

  res /= cub(2.0 * pi);

  return res;
}

double spectra::Pcorr2(Type a, Type b, Type_corr2 c, double k) {
  double xi, wi, qi, xmin, xmax, mumin, mumax, muj, wj, res, integ;

  xmin = kmin / k;
  xmax = kmax / k;

  res = 0.0;

  for (int i = 0; i < nx; i++) {
    gsl_integration_glfixed_point(log(xmin), log(xmax), i, &xi, &wi, t_x);
    xi = exp(xi);
    qi = xi * k;
    mumin = max(-1.0, (1.0 + xi * xi - xmax * xmax) / (2.0 * xi));
    mumax = min(1.0, (1.0 + xi * xi - xmin * xmin) / (2.0 * xi));
    if (xi > 0.5)
      mumax = 0.5 / xi;

    integ = 0.0;
    for (int j = 0; j < nmu; j++) {
      gsl_integration_glfixed_point(mumin, mumax, j, &muj, &wj, t_mu);
      integ += wj * Pcorr2_kernel(a, b, c, k, qi, muj);
    }
    res += wi * cub(qi) * integ * 2.0;
  }

  res /= sqr(2.0 * pi);

  return res;
}

double spectra::Pcorr2_kernel(Type a, Type b, Type_corr2 c, double k, double q,
                              double mu) {
  double kq, res, Gamma;

  kq = sqrt(k * k - 2.0 * mu * k * q + q * q);

  switch (c) {
  case TREE_TREE:
    Gamma = Gamma2_tree(a, kq, q, k) * Gamma2_tree(b, kq, q, k);
    break;
  case TREE_ONELOOP:
    Gamma = Gamma2_tree(a, kq, q, k) * Gamma2_1loop(b, kq, q, k) +
            Gamma2_1loop(a, kq, q, k) * Gamma2_tree(b, kq, q, k);
    break;
  case ONELOOP_ONELOOP:
    Gamma = Gamma2_1loop(a, kq, q, k) * Gamma2_1loop(b, kq, q, k);
    break;
  default:
    Gamma = 0.0;
  }

  res = 2.0 * Gamma * P0(q) * P0(kq);

  return res;
}

double spectra::Pcorr3_kernel(double x[], size_t dim, void *param) {
  double k, p, q, theta1, theta2, phi1, jacobian, res;
  double k_p, k_p_q, prod_q_kp, pklin_p, pklin_q, pklin_kpq;
  double kmin, kmax, pi;
  Pcorr3_integral_params *par;
  gsl_interp_accel *acc;
  gsl_spline *Pspl;
  Type a, b;
  Vector kk, pp, qq, kpq, kp;

  pi = 4.0 * atan(1.0);

  par = (Pcorr3_integral_params *)param;
  k = par->k;
  a = par->a;
  b = par->b;
  kmin = par->kmin;
  kmax = par->kmax;

  p = exp(log(kmin) + (log(kmax) - log(kmin)) * x[0]);
  theta1 = x[1] * pi;
  phi1 = x[2] * 2.0 * pi;
  q = exp(log(kmin) + (log(kmax) - log(kmin)) * x[3]);
  theta2 = x[4] * pi;
  jacobian = sqr((log(kmax) - log(kmin)) * 2.0 * sqr(pi));

  kk.x = 0.0;
  kk.y = 0.0;
  kk.z = k;

  pp.x = p * sin(theta1) * cos(phi1);
  pp.y = p * sin(theta1) * sin(phi1);
  pp.z = p * cos(theta1);

  qq.x = q * sin(theta2);
  qq.y = 0.0;
  qq.z = q * cos(theta2);

  kpq = kk - pp - qq;
  k_p_q = sqrt(kpq * kpq);

  kp = kk - pp;
  k_p = sqrt(kp * kp);
  prod_q_kp = qq * kp;

  if (k_p_q >= kmin && k_p_q <= kmax) {
    if (prod_q_kp <= 0.5 * k_p * k_p) {
      pklin_p = par->cosmo->P0(p);
      pklin_q = par->cosmo->P0(q);
      pklin_kpq = par->cosmo->P0(k_p_q);
      res = F3_sym(a, pp, qq, kpq) * F3_sym(b, pp, qq, kpq) * pklin_p *
            pklin_q * pklin_kpq;
      res *= cub(p * q) * sin(theta1) * sin(theta2);
    } else {
      res = 0.0;
    }
  } else {
    res = 0.0;
  }

  res *= 2.0 * jacobian;

  return res;
}

double spectra::Pcorr3(Type a, Type b, double k) {
  double res, err;
  double xl[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double xu[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
  Pcorr3_integral_params par;
  gsl_monte_function G;

  par.a = a;
  par.b = b;
  par.k = k;
  par.kmin = kmin;
  par.kmax = kmax;
  par.cosmo = cosmo;

  G.f = &(spectra::Pcorr3_kernel);
  G.dim = dim;
  G.params = &par;

  gsl_monte_vegas_init(s);

  /* initial guess */
  gsl_monte_vegas_integrate(&G, xl, xu, dim, MC_calls / 5, r, s, &res, &err);
  do {
    gsl_monte_vegas_integrate(&G, xl, xu, dim, MC_calls, r, s, &res, &err);
    // printf("result = % .6f sigma = % .6f chisq/dof = %.1f\n", res, err,
    // s->chisq);
  } while (fabs(s->chisq - 1.0) > MC_tol);

  res *= 6.0 / sqr(cub(2.0 * pi));

  return res;
}

/* Regularized power spectrum at 1-loop order */
double spectra::Preg_1loop(Type a, Type b, double k) {
  double alpha, Preg1, Preg2;

  alpha = 0.5 * sqr(k) * get_sigmad2(k, lambda) * exp(2.0 * eta);

  Preg1 = exp(2.0 * eta) * exp(-2.0 * alpha) *
          (1.0 + alpha + exp(2.0 * eta) * Gamma1_1loop(a, k)) *
          (1.0 + alpha + exp(2.0 * eta) * Gamma1_1loop(b, k)) * P0(k);

  Preg2 = exp(4.0 * eta) * exp(-2.0 * alpha) * Pcorr2(a, b, TREE_TREE, k);

  return Preg1 + Preg2;
}

/* Regularized power spectrum at 2-loop order */
double spectra::Preg_2loop(Type a, Type b, double k) {
  double alpha, Preg1, Preg2, Preg3;

  alpha = 0.5 * sqr(k) * get_sigmad2(k, lambda) * exp(2.0 * eta);

  Preg1 = exp(2.0 * eta) * exp(-2.0 * alpha) *
          (1.0 + alpha + 0.5 * sqr(alpha) +
           exp(2.0 * eta) * Gamma1_1loop(a, k) * (1.0 + alpha) +
           exp(4.0 * eta) * Gamma1_2loop(a, k)) *
          (1.0 + alpha + 0.5 * sqr(alpha) +
           exp(2.0 * eta) * Gamma1_1loop(b, k) * (1.0 + alpha) +
           exp(4.0 * eta) * Gamma1_2loop(b, k)) *
          P0(k);

  Preg2 = exp(4.0 * eta) * exp(-2.0 * alpha) *
          (Pcorr2(a, b, TREE_TREE, k) * sqr(1.0 + alpha) +
           Pcorr2(a, b, TREE_ONELOOP, k) * exp(2.0 * eta) * (1.0 + alpha) +
           Pcorr2(a, b, ONELOOP_ONELOOP, k) * exp(4.0 * eta));

  Preg3 = exp(6.0 * eta) * exp(-2.0 * alpha) * Pcorr3(a, b, k);

  return Preg1 + Preg2 + Preg3;
}

/* Power spectrum based on 1-loop standard perturbation theory */
double spectra::Pspt_1loop(Type a, Type b, double k) {
  double p0, Pspt1;

  p0 = P0(k);

  Pspt1 = exp(2.0 * eta) * p0 +
          exp(4.0 * eta) * (p0 * Gamma1_1loop(a, k) + p0 * Gamma1_1loop(b, k) +
                            Pcorr2(a, b, TREE_TREE, k));

  return Pspt1;
}

/* Power spectrum based on 2-loop standard perturbation theory */
double spectra::Pspt_2loop(Type a, Type b, double k) {
  double p0, Pspt1, Pspt2;

  p0 = P0(k);

  Pspt1 = exp(2.0 * eta) * p0 +
          exp(4.0 * eta) * (p0 * Gamma1_1loop(a, k) + p0 * Gamma1_1loop(b, k) +
                            Pcorr2(a, b, TREE_TREE, k));

  Pspt2 = exp(6.0 * eta) * (p0 * Gamma1_1loop(a, k) * Gamma1_1loop(b, k) +
                            Pcorr3(a, b, k) + Pcorr2(a, b, TREE_ONELOOP, k) +
                            p0 * Gamma1_2loop(a, k) + p0 * Gamma1_2loop(b, k));

  return Pspt1 + Pspt2;
}

/* output power spectra */
void spectra::output_spectra(void) {
  FILE *fout;
  vector<double> kout;
  double G1_1loop, G1_2loop, P2TREE_TREE, P2TREE_ONELOOP, P2ONELOOP_ONELOOP;
  double P3TREE_TREE, PL, Pnw, Pk;

  if ((fout = fopen(output_fname.c_str(), "w")) == NULL) {
    cerr << "[ERROR] output file open error:" << output_fname << endl;
    exit(1);
  }

  kout.resize(nkout);
  if (spacing == "linear") {
    for (int i = 0; i < nkout; i++) {
      kout[i] = (kmaxout - kminout) / ((double)nkout - 1.0) * i + kminout;
    }
  } else if (spacing == "log") {
    for (int i = 0; i < nkout; i++) {
      kout[i] = (log(kmaxout) - log(kminout)) / ((double)nkout - 1.0) * i +
                log(kminout);
      kout[i] = exp(kout[i]);
    }
  }

  if (model == "diagram") {
    for (int i = 0; i < nkout; i++) {
      G1_1loop = Gamma1_1loop(DENS, kout[i]);
      G1_2loop = Gamma1_2loop(DENS, kout[i]);
      P2TREE_TREE = Pcorr2(DENS, DENS, TREE_TREE, kout[i]);
      P2TREE_ONELOOP = Pcorr2(DENS, DENS, TREE_ONELOOP, kout[i]);
      P2ONELOOP_ONELOOP = Pcorr2(DENS, DENS, ONELOOP_ONELOOP, kout[i]);
      P3TREE_TREE = Pcorr3(DENS, DENS, kout[i]);
      PL = Plin(kout[i]);
      Pnw = Pno_wiggle(kout[i]);

      if (flag_verbose) {
        printf("%g %g %g %g %g %g %g %g %g\n", kout[i], Pnw, PL, G1_1loop,
               G1_2loop, P2TREE_TREE, P2TREE_ONELOOP, P2ONELOOP_ONELOOP,
               P3TREE_TREE);
      }

      fprintf(fout, "%g %g %g %g %g %g %g %g %g\n", kout[i], Pnw, PL, G1_1loop,
              G1_2loop, P2TREE_TREE, P2TREE_ONELOOP, P2ONELOOP_ONELOOP,
              P3TREE_TREE);
    }
  } else {
    if (model == "RegPT" && !flag_1loop) {
      if (flag_verbose)
        printf("# k[h/Mpc] Preg_2loop\n");
      fprintf(fout, "# k[h/Mpc] Preg_2loop\n");
    } else if (model == "RegPT" && flag_1loop) {
      if (flag_verbose)
        printf("# k[h/Mpc] Preg_1loop\n");
      fprintf(fout, "# k[h/Mpc] Preg_1loop\n");
    } else if (model == "SPT" && !flag_1loop) {
      if (flag_verbose)
        printf("# k[h/Mpc] Pspt_2loop\n");
      fprintf(fout, "# k[h/Mpc] Pspt_2loop\n");
    } else if (model == "SPT" && flag_1loop) {
      if (flag_verbose)
        printf("# k[h/Mpc] Pspt_1loop\n");
      fprintf(fout, "# k[h/Mpc] Pspt_1loop\n");
    } else {
      cerr << "[ERROR] model name is invalid" << endl;
      exit(1);
    }

    for (int i = 0; i < nkout; i++) {
      if (model == "RegPT" && !flag_1loop)
        Pk = Preg_2loop(DENS, DENS, kout[i]);
      else if (model == "RegPT" && flag_1loop)
        Pk = Preg_1loop(DENS, DENS, kout[i]);
      else if (model == "SPT" && !flag_1loop)
        Pk = Pspt_2loop(DENS, DENS, kout[i]);
      else if (model == "SPT" && flag_1loop)
        Pk = Pspt_1loop(DENS, DENS, kout[i]);
      else {
        cerr << "[ERROR] model name is invalid" << endl;
        exit(1);
      }
      if (flag_verbose)
        printf("%g %g\n", kout[i], Pk);
      fprintf(fout, "%g %g\n", kout[i], Pk);
    }
  }

  if (flag_verbose) {
    cout << "-> The data is output to \"" << output_fname << "\"." << endl;
  }

  fclose(fout);
}
