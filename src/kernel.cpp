#include "kernel.hpp"
#include "vector.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

inline double sigmaab(int n, Type a, Type b) {
  double nd, res;

  nd = (double)n;
  if (a == DENS && b == DENS)
    res = 2.0 * nd + 1.0;
  else if (a == DENS && b == VELO)
    res = 2.0;
  else if (a == VELO && b == DENS)
    res = 3.0;
  else if (a == VELO && b == VELO)
    res = 2.0 * nd;
  else
    res = 0.0;

  res /= ((2.0 * nd + 3.0) * (nd - 1.0));

  return res;
}

inline double gam_matrix(Type a, Type b, Type c, Vector p, Vector q) {
  double pp, qq, pq, res;

  pp = p * p;
  qq = q * q;
  pq = p * q;

  if (a == DENS && b == DENS && c == VELO)
    res = 1.0 + pq / qq;
  else if (a == DENS && b == VELO && c == DENS)
    res = 1.0 + pq / pp;
  else if (a == VELO && b == VELO && c == VELO)
    res = pq * (pp + qq + 2.0 * pq) / (pp * qq);
  else
    res = 0.0;

  res /= 2.0;

  return res;
}

double F2_sym(Type a, Vector p, Vector q) {
  double pp, qq, mu, res;

  pp = sqrt(p * p);
  qq = sqrt(q * q);
  mu = (p * q) / (pp * qq);

  switch (a) {
  case DENS:
    res = 5.0 / 7.0 + mu / 2.0 * (pp / qq + qq / pp) + 2.0 / 7.0 * sqr(mu);
    break;
  case VELO:
    res = 3.0 / 7.0 + mu / 2.0 * (pp / qq + qq / pp) + 4.0 / 7.0 * sqr(mu);
    break;
  default:
    res = 0.0;
    break;
  }

  return res;
}

inline double F3(Type a, Vector p, Vector q, Vector r) {
  double res;
  Vector qr, pqr;

  qr = q + r;
  pqr = p + qr;

  if ((fabs(qr.x) < EPS && fabs(qr.y) < EPS && fabs(qr.z) < EPS) ||
      (fabs(pqr.x) < EPS && fabs(pqr.y) < EPS && fabs(pqr.z) < EPS)) {
    res = 0.0;
  } else {
    res = (sigmaab(3, a, DENS) * gam_matrix(DENS, DENS, VELO, p, qr) +
           sigmaab(3, a, VELO) * gam_matrix(VELO, VELO, VELO, p, qr)) *
              F2_sym(VELO, q, r) +
          sigmaab(3, a, DENS) * gam_matrix(DENS, VELO, DENS, p, qr) *
              F2_sym(DENS, q, r);
    res *= 2.0;
  }

  return res;
}

double F3_sym(Type a, Vector p, Vector q, Vector r) {
  return (F3(a, p, q, r) + F3(a, r, p, q) + F3(a, q, r, p)) / 3.0;
}

inline double F4a(Type a, Vector p, Vector q, Vector r, Vector s) {
  double res;
  Vector qrs, pqrs;

  qrs = q + r + s;
  pqrs = p + qrs;

  if ((fabs(qrs.x) < EPS && fabs(qrs.y) < EPS && fabs(qrs.z) < EPS) ||
      (fabs(pqrs.x) < EPS && fabs(pqrs.y) < EPS && fabs(pqrs.z) < EPS)) {
    res = 0.0;
  } else {
    res = (sigmaab(4, a, DENS) * gam_matrix(DENS, DENS, VELO, p, qrs) +
           sigmaab(4, a, VELO) * gam_matrix(VELO, VELO, VELO, p, qrs)) *
              F3_sym(VELO, q, r, s) +
          sigmaab(4, a, DENS) * gam_matrix(DENS, VELO, DENS, p, qrs) *
              F3_sym(DENS, q, r, s);
    res *= 2.0;
  }

  return res;
}

inline double F4b(Type a, Vector p, Vector q, Vector r, Vector s) {
  double res;
  Vector pq, rs, pqrs;

  pq = p + q;
  rs = r + s;
  pqrs = pq + rs;

  if ((fabs(pq.x) < EPS && fabs(pq.y) < EPS && fabs(pq.z) < EPS) ||
      (fabs(rs.x) < EPS && fabs(rs.y) < EPS && fabs(rs.z) < EPS) ||
      (fabs(pqrs.x) < EPS && fabs(pqrs.y) < EPS && fabs(pqrs.z) < EPS)) {
    res = 0.0;
  } else {
    res = sigmaab(4, a, DENS) * gam_matrix(DENS, DENS, VELO, pq, rs) *
              F2_sym(DENS, p, q) * F2_sym(VELO, r, s) +
          sigmaab(4, a, DENS) * gam_matrix(DENS, VELO, DENS, pq, rs) *
              F2_sym(VELO, p, q) * F2_sym(DENS, r, s) +
          sigmaab(4, a, VELO) * gam_matrix(VELO, VELO, VELO, pq, rs) *
              F2_sym(VELO, p, q) * F2_sym(VELO, r, s);
  }

  return res;
}

double F4_sym(Type a, Vector p, Vector q, Vector r, Vector s) {
  double res;

  res = (F4a(a, p, q, r, s) + F4a(a, s, p, q, r) + F4a(a, r, s, p, q) +
         F4a(a, q, r, s, p)) /
            4.0 +
        (F4b(a, p, q, r, s) + F4b(a, p, r, q, s) + F4b(a, p, s, q, r) +
         F4b(a, r, s, p, q) + F4b(a, q, s, p, r) + F4b(a, q, r, p, s)) /
            6.0;

  return res;
}

inline double F5a(Type a, Vector p, Vector q, Vector r, Vector s, Vector t) {
  double res;
  Vector qrst, pqrst;

  qrst = q + r + s + t;
  pqrst = p + qrst;

  if ((fabs(qrst.x) < EPS && fabs(qrst.y) < EPS && fabs(qrst.z) < EPS) ||
      (fabs(pqrst.x) < EPS && fabs(pqrst.y) < EPS && fabs(pqrst.z) < EPS)) {
    res = 0.0;
  } else {
    res = (sigmaab(5, a, DENS) * gam_matrix(DENS, DENS, VELO, p, qrst) +
           sigmaab(5, a, VELO) * gam_matrix(VELO, VELO, VELO, p, qrst)) *
              F4_sym(VELO, q, r, s, t) +
          sigmaab(5, a, DENS) * gam_matrix(DENS, VELO, DENS, p, qrst) *
              F4_sym(DENS, q, r, s, t);
    res *= 2.0;
  }

  return res;
}

inline double F5b(Type a, Vector p, Vector q, Vector r, Vector s, Vector t) {
  double res;
  Vector pq, rst, pqrst;

  pq = p + q;
  rst = r + s + t;
  pqrst = pq + rst;

  if ((fabs(pq.x) < EPS && fabs(pq.y) < EPS && fabs(pq.z) < EPS) ||
      (fabs(rst.x) < EPS && fabs(rst.y) < EPS && fabs(rst.z) < EPS) ||
      (fabs(pqrst.x) < EPS && fabs(pqrst.y) < EPS && fabs(pqrst.z) < EPS)) {
    res = 0.0;
  } else {
    res = sigmaab(5, a, DENS) * gam_matrix(DENS, DENS, VELO, pq, rst) *
              F2_sym(DENS, p, q) * F3_sym(VELO, r, s, t) +
          sigmaab(5, a, DENS) * gam_matrix(DENS, VELO, DENS, pq, rst) *
              F2_sym(VELO, p, q) * F3_sym(DENS, r, s, t) +
          sigmaab(5, a, VELO) * gam_matrix(VELO, VELO, VELO, pq, rst) *
              F2_sym(VELO, p, q) * F3_sym(VELO, r, s, t);
    res *= 2.0;
  }

  return res;
}

double F5_sym(Type a, Vector p, Vector q, Vector r, Vector s, Vector t) {
  double res;

  res = (F5a(a, p, q, r, s, t) + F5a(a, t, p, q, r, s) + F5a(a, s, t, p, q, r) +
         F5a(a, r, s, t, p, q) + F5a(a, q, r, s, t, p)) /
            5.0 +
        (F5b(a, p, q, r, s, t) + F5b(a, p, r, q, s, t) + F5b(a, p, s, q, r, t) +
         F5b(a, p, t, q, r, s) + F5b(a, q, r, p, s, t) + F5b(a, q, s, p, r, t) +
         F5b(a, q, t, p, r, s) + F5b(a, r, s, p, q, t) + F5b(a, r, t, p, q, s) +
         F5b(a, s, t, p, q, r)) /
            10.0;

  return res;
}
