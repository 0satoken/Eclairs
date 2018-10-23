#ifndef KERNEL_HEADER_INCLUDED
#define KERNEL_HEADER_INCLUDED

#include <iostream>
#include <cmath>
#include "vector.hpp"

#define EPS (1e-30)

using namespace std;


template<class T> inline T sqr(T x) {return x*x;}
template<class T> inline T cub(T x) {return x*x*x;}
template<class T> inline T qua(T x) {return x*x*x*x;}

enum Type{
  DENS = 1, VELO = 2,
};

double sigmaab(int n, Type a, Type b);
double gam_matrix(Type a, Type b, Type c, Vector p, Vector q);
double F2_sym(Type a, Vector p, Vector q);
double F3(Type a, Vector p, Vector q, Vector r);
double F3_sym(Type a, Vector p, Vector q, Vector r);
double F4a(Type a, Vector p, Vector q, Vector r, Vector s);
double F4b(Type a, Vector p, Vector q, Vector r, Vector s);
double F4_sym(Type a, Vector p, Vector q, Vector r, Vector s);
double F5a(Type a, Vector p, Vector q, Vector r, Vector s, Vector t);
double F5b(Type a, Vector p, Vector q, Vector r, Vector s, Vector t);
double F5_sym(Type a, Vector p, Vector q, Vector r, Vector s, Vector t);

double kernel_Gamma1_1loop(Type a, double k, double q);
double kernel_Gamma1_2loop(Type a, double k, double q1, double q2);
double alpha(Type a, double q1, double q2);
double beta(Type a, double q1, double q2);
double cc(Type a, double qs);
double dd(Type a, double qr);
double delta(Type a, double qs);

double kernel_Gamma2_1loop(Type a, double k1, double k2, double k3, double q);
double kernel_Gamma2d_reg3rd(double k1, double k2, double k3, double q);
double kernel_Gamma2d_highk(double k1, double k2, double k3, double q);
double kernel_Gamma2d_lowk(double k1, double k2, double k3, double q);
double kernel_Gamma2d_exact(double k1, double k2, double k3, double q);
double kernel_Gamma2d_iso(double k1, double k2, double k3, double q);
double kernel_Gamma2d_elongate(double k1, double k2, double k3, double q);
double kernel_Gamma2d_coll(double k1, double k2, double k3, double q);
double kernel_Gamma2v_reg3rd(double k1, double k2, double k3, double q);
double kernel_Gamma2v_highk(double k1, double k2, double k3, double q);
double kernel_Gamma2v_lowk(double k1, double k2, double k3, double q);
double kernel_Gamma2v_exact(double k1, double k2, double k3, double q);
double kernel_Gamma2v_iso(double k1, double k2, double k3, double q);
double kernel_Gamma2v_elongate(double k1, double k2, double k3, double q);
double kernel_Gamma2v_coll(double k1, double k2, double k3, double q);
inline double LFunc(double k, double q);
double WFunc(double k1, double k2, double k3, double q);
double betafunc(int i, double z);
double small_beta(double k, double q);
double big_beta(double k1, double k2, double k3, double q);

#endif
