#ifndef VECTOR_HEADER_INCLUDED
#define VECTOR_HEADER_INCLUDED

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>


class Vector{
public:
  Vector();
  Vector(double _x, double _y, double _z);
  Vector& operator=(const Vector& v);
  Vector& operator+=(const Vector& v);
  Vector& operator-=(const Vector& v);
  Vector& operator*=(double k);
  Vector& operator/=(double k);
  Vector operator+();
  Vector operator-();

  double x;
  double y;
  double z;
};

Vector operator+(const Vector& u, const Vector& v);
Vector operator-(const Vector& u, const Vector& v);
double operator*(const Vector& u, const Vector& v);
Vector operator*(const Vector& v, double k);
Vector operator*(double k, const Vector& v);
Vector operator/(const Vector& v, double k);

#endif
