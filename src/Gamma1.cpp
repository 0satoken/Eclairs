#include "kernel.hpp"
#include "vector.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

/*
 * kernel function for Gamma_1^(1-loop) and Gamma_1^(2-loop)
 * Eq.21 and 22 of Bernardeau, Taruya, Nishimichi, PRD, 89, 023502 (2014)
 * This kernel can be reduced as the dimensionless function of x = q/k.
 */
double kernel_Gamma1_1loop(Type a, double k, double q) {
  double x, res;

  x = q / k;
  res = 0.0;

  switch (a) {
  case DENS:
    if (x < 10.0 && fabs(x - 1.0) > 0.01) {
      res = 6.0 / (x * x) - 79.0 + 50.0 * x * x - 21.0 * qua(x) +
            0.75 * cub(1.0 / x - x) * (2.0 + 7.0 * x * x) *
                log(sqr(fabs((1.0 - x) / (1.0 + x))));
      res /= 504.0;
    } else if (fabs(x - 1.0) <= 0.01) {
      res = -11.0 / 126.0 + (x - 1.0) / 126.0 - 29.0 / 252.0 * sqr(x - 1.0);
    } else if (x >= 10.0) {
      res = -61.0 / 630.0 + 2.0 / 105.0 / (x * x) - 10.0 / 1323.0 / qua(x);
    }
    res /= x * x;
    break;
  case VELO:
    if (x < 10.0 && fabs(x - 1.0) > 0.01) {
      res = 6.0 / (x * x) - 41.0 + 2.0 * x * x - 3.0 * qua(x) +
            0.750 * cub(1.0 / x - x) * (2.0 + x * x) *
                log(sqr(fabs((1.0 - x) / (1.0 + x))));
      res /= 168.0;
    } else if (fabs(x - 1.0) <= 0.01) {
      res = -3.0 / 14.0 - 5.0 / 42.0 * (x - 1.0) + sqr(x - 1.0) / 84.0;
    } else if (x >= 10.0) {
      res = -3.0 / 10.0 + 26.0 / 245.0 / (x * x) - 38.0 / 2205.0 / qua(x);
    }
    res /= x * x;
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}

double kernel_Gamma1_2loop(Type a, double k, double q1, double q2) {
  double x1, x2, res;

  x1 = q1 / k;
  x2 = q2 / k;

  res = -alpha(a, x1, x2) / (x1 * x1 + x2 * x2);

  /* The original form is below. But for the first term, the interval in
  the integral has to be taken short. This integration is carried out
  spearately by the function kernel_Gamma1_1loop */
  /*
  res = 0.5*kernel_Gamma1_1loop(a, k, q1) * kernel_Gamma1_1loop(a, k, q2)
      - alpha(a, x1, x2)/(x1*x1+x2*x2);
  */

  return res;
}

double alpha(Type a, double q1, double q2) {
  double qr, qs;

  qr = sqrt(q1 * q1 + q2 * q2);
  qs = q1 * q2 / qr;

  return (beta(a, q1, q2) - cc(a, qs) - dd(a, qr)) * delta(a, qs);
}

double beta(Type a, double q1, double q2) {
  double res, y;

  y = (q1 - q2) / (q1 + q2);

  switch (a) {
  case DENS:
    if (fabs(fabs(y) - 1.0) < 0.05) {
      res = 120424.0 / 3009825.0 + (2792.0 * sqr(1.0 - fabs(y))) / 429975.0 +
            (2792.0 * cub(1.0 - fabs(y))) / 429975.0 +
            (392606.0 * qua(1.0 - fabs(y))) / 99324225.0;
    } else if (fabs(y) < 0.01) {
      res = 22382.0 / 429975.0 - 57052.0 * sqr(y) / 429975.0;
    } else {
      res = (2.0 * (1.0 + sqr(y)) *
             (-11191.0 + 118054.0 * sqr(y) - 18215.0 * qua(y) +
              18215.0 * sqr(qua(y)) - 118054.0 * (y * cub(cub(y))) +
              11191.0 * qua(cub(y)) +
              60.0 * qua(y) * (3467.0 - 790.0 * sqr(y) + 3467.0 * qua(y)) *
                  log(sqr(y)))) /
            (429975.0 * cub(-1.0 + sqr(y)) * qua(-1.0 + sqr(y)));
    }
    break;
  case VELO:
    if (fabs(fabs(y) - 1.0) < 0.05) {
      res = 594232.0 / 33108075.0 +
            (91912.0 * sqr(1.0 - fabs(y))) / 33108075.0 -
            (91912.0 * cub(1.0 - fabs(y))) / 33108075.0 +
            (1818458.0 * qua(1.0 - fabs(y))) / 1092566475.0;
    } else if (fabs(y) < 0.01) {
      res = 9886.0 / 429975.0 - 254356.0 * sqr(y) / 4729725.0;
    } else {
      res = (2.0 * (1.0 + sqr(y)) *
             (-54373.0 + 562162.0 * sqr(y) - 408245.0 * qua(y) +
              408245.0 * sqr(qua(y)) - 562162.0 * (y * cub(cub(y))) +
              54373.0 * qua(cub(y)) +
              60.0 * qua(y) * (14561.0 - 10690.0 * sqr(y) + 14561.0 * qua(y)) *
                  log(sqr(y)))) /
            (4729725.0 * cub(-1.0 + sqr(y)) * qua(-1.0 + sqr(y)));
    }
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}

double cc(Type a, double qs) {
  double res;

  switch (a) {
  case DENS:
    res =
        0.02088557981734545 /
            (35.09866396385646 * qua(qs) + 4.133811416743832 * sqr(qs) + 1.0) -
        0.076100391588544 * cub(qs) / (77.79670692480381 * sqr(cub(qs)) + 1.0);
    break;
  case VELO:
    res = -0.008217140060512867 / (42.14072830553836 * qua(qs) +
                                   1.367564560397748 * sqr(qs) + 1.0) +
          0.01099093588476197 * cub(qs) /
              (28.490424851390667 * sqr(qua(qs)) + 1.0);
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}

double dd(Type a, double qr) {
  double res;

  switch (a) {
  case DENS:
    res = -0.022168478217299517 / (7.030631093970638 * qua(qr) +
                                   2.457866449142683 * sqr(qr) + 1.0) +
          0.009267495321465601 * sqr(qr) /
              (4.11633699497035 * (qr * cub(cub(qr))) + 1.0);
    break;
  case VELO:
    res = 0.008023147297149955 / (2.238261369090066 * sqr(qr) * cub(qr) + 1.0) -
          0.006173880966928251 * sqr(qr) /
              (0.4711737436482179 * sqr(qr) * cub(qr) + 1.0);
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}

double delta(Type a, double qs) {
  double res, qs_;

  qs_ = sqrt(2.0) * qs;

  switch (a) {
  case DENS:
    res =
        0.3191221482038663 * qua(qs_) / (1.3549058352752525 * qua(qs_) + 1.0) +
        1.2805575495849764 / (18.192939946270577 * qua(qs_) +
                              3.98817716852858 * sqr(qs_) + 1.0) +
        0.764469131436698;
    break;
  case VELO:
    res = 1.528058751211026 *
              (2.4414566000839355 * qua(qs_) + 1.8616263354608626 * sqr(qs_)) /
              (2.4414566000839355 * qua(qs_) + 1.0) +
          2.5227965281961247 / (0.0028106312591877226 * qua(qs_) +
                                1.0332351481570086 * sqr(qs_) + 1.0) -
          0.528058751211026;
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}
