#include "IR_EFT.hpp"
#include "cosmology.hpp"
#include "direct_red.hpp"
#include "fast_spectra.hpp"
#include "kernel.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "spectra_red.hpp"
#include "vector.hpp"
#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace p = boost::python;
namespace np = boost::python::numpy;

/* This function coverts vector to ndarray */
np::ndarray vec_to_ndarray(vector<double> &vec) {
  size_t size;

  size = vec.size();
  np::dtype dt = np::dtype::get_builtin<double>();
  p::tuple shape = p::make_tuple(size);
  np::ndarray a = np::zeros(shape, dt);

  for (size_t i = 0; i < size; ++i)
    a[i] = vec[i];

  return a;
}

/* This function coverts vector to ndarray */
np::ndarray vec_to_ndarray(vector<vector<double>> &vec) {
  size_t size1, size2;

  size1 = vec.size();
  size2 = vec[0].size();
  np::dtype dt = np::dtype::get_builtin<double>();
  p::tuple shape = p::make_tuple(size1, size2);
  np::ndarray a = np::zeros(shape, dt);

  for (size_t i = 0; i < size1; ++i) {
    for (size_t j = 0; j < size2; ++j)
      a[i][j] = vec[i][j];
  }

  return a;
}

/* This function checks the given array is compatible */
void check_array(np::ndarray a) {
  if (a.get_nd() != 1) {
    cerr << "[ERROR] array must be 1-dimensional" << endl;
    exit(1);
  }

  if (a.get_dtype() != np::dtype::get_builtin<double>()) {
    cerr << "[ERROR] array must be float64 array" << endl;
    exit(1);
  }

  return;
}

class pyeclairs {
private:
  params Params;
  cosmology *Cosmo;
  spectra *Spectra;
  direct_red *Direct_red;
  fast_spectra *Fast_spectra;
  spectra_red *Spectra_red;
  IR_EFT *Ir_eft;
  map<string, bool> pyflags;

public:
  pyeclairs();
  ~pyeclairs();
  void initialize(p::dict &py_dict, np::ndarray k, np::ndarray Tk);
  np::ndarray get_Plinear(np::ndarray k);
  np::ndarray get_Pnowiggle(np::ndarray k);
  p::dict get_spectra_2l(np::ndarray k, p::list type, string mode = "RegPT");
  p::dict get_spectra_1l(np::ndarray k, p::list type, string mode = "RegPT");
  p::dict get_direct_Aterm(np::ndarray k);
  p::dict get_direct_Bterm(np::ndarray k);
  p::dict get_fast_spectra_2l(np::ndarray k, p::list type);
  p::dict get_fast_Aterm(np::ndarray k);
  p::dict get_fast_Bterm(np::ndarray k);
  np::ndarray get_2D_power(np::ndarray k, np::ndarray mu);
  np::ndarray get_multipoles(np::ndarray k, np::ndarray l);
  np::ndarray get_wedges(np::ndarray k, np::ndarray w);
  p::dict get_multipoles_grid(np::ndarray kbin, np::ndarray l);
  p::dict get_wedges_grid(np::ndarray kbin, np::ndarray w);
};

pyeclairs::pyeclairs() {}

pyeclairs::~pyeclairs() {
  delete Cosmo;
  delete Spectra;
  if (pyflags["IREFT_mode"]) {
    delete Ir_eft;
  }
  if (pyflags["fast_mode"]) {
    delete Fast_spectra;
    delete Spectra_red;
  }
  if (pyflags["direct_mode"]) {
    delete Direct_red;
    delete Spectra_red;
  }
}

void pyeclairs::initialize(p::dict &py_dict, np::ndarray k, np::ndarray Tk) {
  int N, cnt;
  vector<double> k_set, Tk_set;
  vector<string> name_list;
  map<string, TYPE> type_list;
  string key;

  name_list = Params.get_name_list();
  type_list = Params.get_type_list();

  check_array(k);
  check_array(Tk);

  if (k.shape(0) != Tk.shape(0)) {
    cerr << "[ERROR] The length of k and Tk should be the same." << endl;
    exit(1);
  }

  N = k.shape(0);

  for (int i = 0; i < N; ++i) {
    k_set.push_back(p::extract<double>(k[i]));
    Tk_set.push_back(p::extract<double>(Tk[i]));
  }

  for (size_t i = 0; i < name_list.size(); ++i) {
    key = name_list[i];
    TYPE type = type_list[key];

    if (!py_dict.has_key(key))
      continue;

    if (type == INT) {
      p::extract<int> extracted_vali(py_dict[key]);
      Params.iparams[key] = extracted_vali;
    } else if (type == DOUBLE) {
      p::extract<double> extracted_vald(py_dict[key]);
      Params.dparams[key] = extracted_vald;
    } else if (type == STRING) {
      p::extract<string> extracted_vals(py_dict[key]);
      Params.sparams[key] = extracted_vals;
    } else if (type == BOOL) {
      p::extract<bool> extracted_valb(py_dict[key]);
      Params.bparams[key] = extracted_valb;
    }
  }

  Params.show_parameter();

  pyflags["fast_mode"] = Params.bparams["fast_mode"];
  pyflags["direct_mode"] = Params.bparams["direct_mode"];
  pyflags["IREFT_mode"] = Params.bparams["IREFT_mode"];

  cnt = 0;
  cnt += (pyflags["fast_mode"]) ? (1) : (0);
  cnt += (pyflags["direct_mode"]) ? (1) : (0);
  cnt += (pyflags["IREFT_mode"]) ? (1) : (0);

  if (cnt != 1) {
    cerr << "[ERROR] multiple or no modes are set" << endl;
    exit(1);
  }

  Cosmo = new cosmology(Params);
  Cosmo->set_transfer(k_set, Tk_set);
  Cosmo->set_spectra();

  Spectra = new spectra(Params, *Cosmo);

  if (pyflags["IREFT_mode"]) {
    cout << "[NOTE] IR EFT mode" << endl;
    Ir_eft = new IR_EFT(Params, *Cosmo, *Spectra);
  }

  if (pyflags["fast_mode"]) {
    cout << "[NOTE] Fast mode" << endl;
    Fast_spectra = new fast_spectra(Params, *Cosmo, *Spectra);
    Spectra_red = new spectra_red(Params, *Cosmo, *Fast_spectra);
  }

  if (pyflags["direct_mode"]) {
    cout << "[NOTE] Direct mode" << endl;
    Direct_red = new direct_red(Params, *Cosmo, *Spectra);
    Spectra_red = new spectra_red(Params, *Cosmo, *Direct_red);
  }

  return;
}

np::ndarray pyeclairs::get_Plinear(np::ndarray k) {
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for (int i = 0; i < N; ++i) {
    ki = p::extract<double>(k[i]);
    Pi = Cosmo->Plin(ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyeclairs::get_Pnowiggle(np::ndarray k) {
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for (int i = 0; i < N; ++i) {
    ki = p::extract<double>(k[i]);
    Pi = Cosmo->Pno_wiggle(ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

p::dict pyeclairs::get_spectra_2l(np::ndarray k, p::list type, string mode) {
  double ki, Pi;
  string type0;
  size_t N;
  vector<double> res_dd, res_dt, res_tt;
  p::dict P;

  if (!(mode == "RegPT" || mode == "SPT")) {
    cerr << "[ERROR] invalid mode: " << mode << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  for (int i = 0; i < len(type); ++i) {
    type0 = p::extract<std::string>(type[i]);
    if (type0 == "dd") {
      for (int ik = 0; ik < N; ++ik) {
        ki = p::extract<double>(k[ik]);
        if (mode == "RegPT")
          Pi = Spectra->Preg_2loop(DENS, DENS, ki);
        else if (mode == "SPT")
          Pi = Spectra->Pspt_2loop(DENS, DENS, ki);
        res_dd.push_back(Pi);
      }
      P[type0] = vec_to_ndarray(res_dd);
    } else if (type0 == "dt") {
      for (int ik = 0; ik < N; ++ik) {
        ki = p::extract<double>(k[ik]);
        if (mode == "RegPT")
          Pi = Spectra->Preg_2loop(DENS, VELO, ki);
        else if (mode == "SPT")
          Pi = Spectra->Pspt_2loop(DENS, VELO, ki);
        res_dt.push_back(Pi);
      }
      P[type0] = vec_to_ndarray(res_dt);
    } else if (type0 == "tt") {
      for (int ik = 0; ik < N; ++ik) {
        ki = p::extract<double>(k[ik]);
        if (mode == "RegPT")
          Pi = Spectra->Preg_2loop(VELO, VELO, ki);
        else if (mode == "SPT")
          Pi = Spectra->Pspt_2loop(VELO, VELO, ki);
        res_tt.push_back(Pi);
      }
      P[type0] = vec_to_ndarray(res_tt);
    } else {
      cerr << "[ERROR] invalid type: " << type0 << endl;
      exit(1);
    }
  }

  return P;
}

p::dict pyeclairs::get_spectra_1l(np::ndarray k, p::list type, string mode) {
  double ki, Pi;
  string type0;
  size_t N;
  vector<double> res_dd, res_dt, res_tt;
  p::dict P;

  if (!(mode == "RegPT" || mode == "SPT")) {
    cerr << "[ERROR] invalid mode: " << mode << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  for (int i = 0; i < len(type); ++i) {
    type0 = p::extract<std::string>(type[i]);
    if (type0 == "dd") {
      for (int ik = 0; ik < N; ++ik) {
        ki = p::extract<double>(k[ik]);
        if (mode == "RegPT")
          Pi = Spectra->Preg_1loop(DENS, DENS, ki);
        else if (mode == "SPT")
          Pi = Spectra->Pspt_1loop(DENS, DENS, ki);
        res_dd.push_back(Pi);
      }
      P[type0] = vec_to_ndarray(res_dd);
    } else if (type0 == "dt") {
      for (int ik = 0; ik < N; ++ik) {
        ki = p::extract<double>(k[ik]);
        if (mode == "RegPT")
          Pi = Spectra->Preg_1loop(DENS, VELO, ki);
        else if (mode == "SPT")
          Pi = Spectra->Pspt_1loop(DENS, VELO, ki);
        res_dt.push_back(Pi);
      }
      P[type0] = vec_to_ndarray(res_dt);
    } else if (type0 == "tt") {
      for (int ik = 0; ik < N; ++ik) {
        ki = p::extract<double>(k[ik]);
        if (mode == "RegPT")
          Pi = Spectra->Preg_1loop(VELO, VELO, ki);
        else if (mode == "SPT")
          Pi = Spectra->Pspt_1loop(VELO, VELO, ki);
        res_tt.push_back(Pi);
      }
      P[type0] = vec_to_ndarray(res_tt);
    } else {
      cerr << "[ERROR] invalid type: " << type0 << endl;
      exit(1);
    }
  }

  return P;
}

p::dict pyeclairs::get_direct_Aterm(np::ndarray k) {
  size_t N;
  vector<double> k0;
  map<string, vector<double>> res;
  p::dict A;

  if (!pyflags["direct_mode"]) {
    cerr << "[ERROR] direct mode is off" << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  k0.resize(N);

  for (int ik = 0; ik < N; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  res = Direct_red->get_Aterm(k0);

  A["A2"] = vec_to_ndarray(res["A2"]);
  A["A4"] = vec_to_ndarray(res["A4"]);
  A["A6"] = vec_to_ndarray(res["A6"]);

  return A;
}

p::dict pyeclairs::get_direct_Bterm(np::ndarray k) {
  size_t N;
  vector<double> k0;
  map<string, vector<double>> res;
  p::dict B;

  if (!pyflags["direct_mode"]) {
    cerr << "[ERROR] direct mode is off" << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  k0.resize(N);

  for (int ik = 0; ik < N; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  res = Direct_red->get_Bterm(k0);

  B["B2"] = vec_to_ndarray(res["B2"]);
  B["B4"] = vec_to_ndarray(res["B4"]);
  B["B6"] = vec_to_ndarray(res["B6"]);
  B["B8"] = vec_to_ndarray(res["B8"]);

  return B;
}

p::dict pyeclairs::get_fast_spectra_2l(np::ndarray k, p::list type) {
  size_t N;
  string type0;
  vector<double> k0;
  map<string, vector<double>> res;
  p::dict P;

  if (!pyflags["fast_mode"]) {
    cerr << "[ERROR] fast mode is off" << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  k0.resize(N);

  for (int ik = 0; ik < N; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  res = Fast_spectra->get_spectra_2l(k0);

  for (int i = 0; i < len(type); ++i) {
    type0 = p::extract<std::string>(type[i]);
    if (type0 == "dd" || type0 == "dt" || type0 == "tt") {
      P[type0] = vec_to_ndarray(res[type0]);
    } else {
      cerr << "[ERROR] invalid type: " << type0 << endl;
      exit(1);
    }
  }

  return P;
}

p::dict pyeclairs::get_fast_Aterm(np::ndarray k) {
  size_t N;
  vector<double> k0;
  map<string, vector<double>> res;
  p::dict A;

  if (!pyflags["fast_mode"]) {
    cerr << "[ERROR] fast mode is off" << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  k0.resize(N);

  for (int ik = 0; ik < N; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  res = Fast_spectra->get_Aterm(k0);

  A["A2"] = vec_to_ndarray(res["A2"]);
  A["A4"] = vec_to_ndarray(res["A4"]);
  A["A6"] = vec_to_ndarray(res["A6"]);

  return A;
}

p::dict pyeclairs::get_fast_Bterm(np::ndarray k) {
  size_t N;
  vector<double> k0;
  map<string, vector<double>> res;
  p::dict B;

  if (!pyflags["fast_mode"]) {
    cerr << "[ERROR] fast mode is off" << endl;
    exit(1);
  }

  check_array(k);
  N = k.shape(0);

  k0.resize(N);

  for (int ik = 0; ik < N; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  res = Fast_spectra->get_Bterm(k0);

  B["B2"] = vec_to_ndarray(res["B2"]);
  B["B4"] = vec_to_ndarray(res["B4"]);
  B["B6"] = vec_to_ndarray(res["B6"]);
  B["B8"] = vec_to_ndarray(res["B8"]);

  return B;
}

np::ndarray pyeclairs::get_2D_power(np::ndarray k, np::ndarray mu) {
  size_t Nk, Nmu;
  vector<double> k0, mu0;
  vector<vector<double>> res;

  check_array(k);
  Nk = k.shape(0);

  k0.resize(Nk);

  for (int ik = 0; ik < Nk; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  check_array(mu);
  Nmu = mu.shape(0);

  mu0.resize(Nmu);

  for (int imu = 0; imu < Nmu; ++imu) {
    mu0[imu] = p::extract<double>(mu[imu]);
  }

  if (pyflags["fast_mode"] || pyflags["direct_mode"]) {
    res = Spectra_red->get_2D_power(k0, mu0);
  }

  if (pyflags["IREFT_mode"]) {
    res = Ir_eft->calc_Pkmu_1l(k0, mu0);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyeclairs::get_multipoles(np::ndarray k, np::ndarray l) {
  size_t Nk, Nl;
  vector<double> k0;
  vector<int> l0;
  vector<vector<double>> res;

  check_array(k);
  Nk = k.shape(0);

  k0.resize(Nk);

  for (int ik = 0; ik < Nk; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  Nl = l.shape(0);

  l0.resize(Nl);

  for (int il = 0; il < Nl; ++il) {
    l0[il] = (int)p::extract<int64_t>(l[il]);
  }

  if (pyflags["fast_mode"] || pyflags["direct_mode"]) {
    res = Spectra_red->get_multipoles(k0, l0);
  }

  if (pyflags["IREFT_mode"]) {
    res = Ir_eft->get_multipoles(k0, l0);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyeclairs::get_wedges(np::ndarray k, np::ndarray w) {
  size_t Nk, Nw;
  vector<double> k0;
  vector<double> w0;
  vector<vector<double>> res;

  check_array(k);
  Nk = k.shape(0);

  k0.resize(Nk);

  for (int ik = 0; ik < Nk; ++ik) {
    k0[ik] = p::extract<double>(k[ik]);
  }

  Nw = w.shape(0);

  w0.resize(Nw);

  for (int iw = 0; iw < Nw; ++iw) {
    w0[iw] = p::extract<double>(w[iw]);
  }

  if (pyflags["fast_mode"] || pyflags["direct_mode"]) {
    res = Spectra_red->get_wedges(k0, w0);
  }

  if (pyflags["IREFT_mode"]) {
    res = Ir_eft->get_wedges(k0, w0);
  }

  return vec_to_ndarray(res);
}

p::dict pyeclairs::get_multipoles_grid(np::ndarray kbin, np::ndarray l) {
  size_t Nkbin, Nl;
  vector<double> kbin0;
  vector<int> l0;
  pair<vector<double>, vector<vector<double>>> res;
  p::dict return_values;

  check_array(kbin);
  Nkbin = kbin.shape(0);
  kbin0.resize(Nkbin);

  for (int ikbin = 0; ikbin < Nkbin; ++ikbin) {
    kbin0[ikbin] = p::extract<double>(kbin[ikbin]);
  }

  Nl = l.shape(0);
  l0.resize(Nl);

  for (int il = 0; il < Nl; ++il) {
    l0[il] = (int)p::extract<int64_t>(l[il]);
  }

  if (pyflags["fast_mode"] || pyflags["direct_mode"]) {
    res = Spectra_red->get_multipoles_grid(kbin0, l0);
  }

  if (pyflags["IREFT_mode"]) {
    res = Ir_eft->get_multipoles_grid(kbin0, l0);
  }

  return_values["kmean"] = vec_to_ndarray(res.first);
  return_values["multipoles"] = vec_to_ndarray(res.second);

  return return_values;
}

p::dict pyeclairs::get_wedges_grid(np::ndarray kbin, np::ndarray w) {
  size_t Nkbin, Nw;
  vector<double> kbin0;
  vector<double> w0;
  tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> res;
  p::dict return_values;

  check_array(kbin);
  Nkbin = kbin.shape(0);
  kbin0.resize(Nkbin);

  for (int ikbin = 0; ikbin < Nkbin; ++ikbin) {
    kbin0[ikbin] = p::extract<double>(kbin[ikbin]);
  }

  Nw = w.shape(0);
  w0.resize(Nw);

  for (int iw = 0; iw < Nw; ++iw) {
    w0[iw] = p::extract<double>(w[iw]);
  }

  if (pyflags["fast_mode"] || pyflags["direct_mode"]) {
    res = Spectra_red->get_wedges_grid(kbin0, w0);
  }

  if (pyflags["IREFT_mode"]) {
    res = Ir_eft->get_wedges_grid(kbin0, w0);
  }

  return_values["kmean"] = vec_to_ndarray(get<0>(res));
  return_values["mumean"] = vec_to_ndarray(get<1>(res));
  return_values["wedges"] = vec_to_ndarray(get<2>(res));

  return return_values;
}

/* wrappers for overloads */
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_spectra_2l_ol, get_spectra_2l, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_spectra_1l_ol, get_spectra_1l, 2, 3)

BOOST_PYTHON_MODULE(pyeclairs) {
  np::initialize();

  p::class_<pyeclairs>("pyeclairs")
      .def("initialize", &pyeclairs::initialize)
      .def("get_Plinear", &pyeclairs::get_Plinear)
      .def("get_Pnowiggle", &pyeclairs::get_Pnowiggle)
      .def("get_spectra_2l", &pyeclairs::get_spectra_2l, get_spectra_2l_ol())
      .def("get_spectra_1l", &pyeclairs::get_spectra_1l, get_spectra_1l_ol())
      .def("get_direct_Aterm", &pyeclairs::get_direct_Aterm)
      .def("get_direct_Bterm", &pyeclairs::get_direct_Bterm)
      .def("get_fast_spectra_2l", &pyeclairs::get_fast_spectra_2l)
      .def("get_fast_Aterm", &pyeclairs::get_fast_Aterm)
      .def("get_fast_Bterm", &pyeclairs::get_fast_Bterm)
      .def("get_2D_power", &pyeclairs::get_2D_power)
      .def("get_multipoles", &pyeclairs::get_multipoles)
      .def("get_wedges", &pyeclairs::get_wedges)
      .def("get_multipoles_grid", &pyeclairs::get_multipoles_grid)
      .def("get_wedges_grid", &pyeclairs::get_wedges_grid);
}
