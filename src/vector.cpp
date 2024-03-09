#include "vector.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

Vector::Vector() : x(0), y(0), z(0) {}

Vector::Vector(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

Vector &Vector::operator=(const Vector &v) {
  this->x = v.x;
  this->y = v.y;
  this->z = v.z;
  return *this;
}

Vector &Vector::operator+=(const Vector &v) {
  this->x += v.x;
  this->y += v.y;
  this->z += v.z;
  return *this;
}

Vector &Vector::operator-=(const Vector &v) {
  this->x -= v.x;
  this->y -= v.y;
  this->z -= v.z;
  return *this;
}

Vector &Vector::operator*=(double k) {
  this->x *= k;
  this->y *= k;
  this->z *= k;
  return *this;
}

Vector &Vector::operator/=(double k) {
  this->x /= k;
  this->y /= k;
  this->z /= k;
  return *this;
}

Vector Vector::operator+() { return *this; }

Vector Vector::operator-() { return Vector(-x, -y, -z); }

Vector operator+(const Vector &u, const Vector &v) {
  Vector w;
  w.x = u.x + v.x;
  w.y = u.y + v.y;
  w.z = u.z + v.z;
  return w;
}

Vector operator-(const Vector &u, const Vector &v) {
  Vector w;
  w.x = u.x - v.x;
  w.y = u.y - v.y;
  w.z = u.z - v.z;
  return w;
}

double operator*(const Vector &u, const Vector &v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

Vector operator*(const Vector &v, double k) {
  Vector w;
  w.x = v.x * k;
  w.y = v.y * k;
  w.z = v.z * k;
  return w;
}

Vector operator*(double k, const Vector &v) {
  Vector w;
  w.x = v.x * k;
  w.y = v.y * k;
  w.z = v.z * k;
  return w;
}

Vector operator/(const Vector &v, double k) {
  Vector w;
  w.x = v.x / k;
  w.y = v.y / k;
  w.z = v.z / k;
  return w;
}
