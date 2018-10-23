#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "kernel.hpp"
#include "vector.hpp"
#include "io.hpp"
#include "cosmology.hpp"
#include "spectra.hpp"


namespace p = boost::python;
namespace np = boost::python::numpy;

/* This function coverts vector to ndarray */
template<class T>
np::ndarray vec_to_ndarray(vector<T> &vec){
    size_t size;

    size = vec.size();
    np::dtype dt = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(size);
    np::ndarray a = np::zeros(shape, dt);


    for(int i=0;i<size;++i) a[i] = vec[i];

    return a;
}

/* This function checks the given array is compatible */
void check_array(np::ndarray a){
    if(a.get_nd() != 1){
        cerr << "[ERROR] array must be 1-dimensional" << endl;
        exit(1);
    }

    if(a.get_dtype() != np::dtype::get_builtin<double>()){
        cerr << "[ERROR] array must be float64 array" << endl;
        exit(1);
    }

    return;
}

class pyspectrum
{
private:
  cosmology *Cosmo;
  spectra *Spectra;
  params Params;
public:
  void set_cosmology(p::dict &py_dict, np::ndarray k, np::ndarray Tk);
  np::ndarray calc_linear(np::ndarray k);
  np::ndarray calc_no_wiggle(np::ndarray k);
  np::ndarray calc_RegPT_2loop(np::ndarray k);
  np::ndarray calc_SPT_2loop(np::ndarray k);
  np::ndarray calc_RegPT_1loop(np::ndarray k);
  np::ndarray calc_SPT_1loop(np::ndarray k);
};


void pyspectrum::set_cosmology(p::dict &py_dict, np::ndarray k, np::ndarray Tk){
  int N;
  bool flag_smoothing = false;
  vector<double> k_set, Tk_set;


  check_array(k);
  check_array(Tk);

  if(k.shape(0) != Tk.shape(0)){
      cerr << "[ERROR] The length of k and Tk should be the same." << endl;
      exit(1);
  }

  N = k.shape(0);

  for(int i=0;i<N;++i){
    k_set.push_back(p::extract<double>(k[i]));
    Tk_set.push_back(p::extract<double>(Tk[i]));
  }

  p::list keys = py_dict.keys();
  for(int i=0;i<len(keys);++i){
    p::extract<string> extracted_key(keys[i]);
    if(!extracted_key.check()){
      cout << "[NOTE] Key invalid, map might be incomplete" << endl;
      continue;
    }

    string key = extracted_key;
    if(key == "smoothing"){
      p::extract<bool> extracted_val(py_dict[key]);
      if(extracted_val) flag_smoothing = true;
    }
    else{
      p::extract<double> extracted_val(py_dict[key]);
      if(!extracted_val.check()){
        cout << "[NOTE] Value invalid, map might be incomplete" << endl;
        continue;
      }
      double value = extracted_val;
      Params.dparams[key] = value;
    }
  }


  Params.bparams["output"] = false; // no output

  Cosmo = new cosmology(Params);
  Cosmo->set_transfer(k_set, Tk_set);
  Cosmo->set_spectra();
  if(flag_smoothing){
    cout << "[NOTE] Smoothed linear power spectrum is set." << endl;
    Cosmo->set_smoothed_spectra();
  }

  Spectra = new spectra(*Cosmo);

  return;
}

np::ndarray pyspectrum::calc_linear(np::ndarray k){
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for(int i=0;i<N;++i){
    ki = p::extract<double>(k[i]);
    Pi = Spectra->Plin(ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyspectrum::calc_no_wiggle(np::ndarray k){
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for(int i=0;i<N;++i){
    ki = p::extract<double>(k[i]);
    Pi = Spectra->Pno_wiggle(ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyspectrum::calc_RegPT_2loop(np::ndarray k){
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for(int i=0;i<N;++i){
    ki = p::extract<double>(k[i]);
    Pi = Spectra->Preg_2loop(DENS, DENS, ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyspectrum::calc_SPT_2loop(np::ndarray k){
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for(int i=0;i<N;++i){
    ki = p::extract<double>(k[i]);
    Pi = Spectra->Pspt_2loop(DENS, DENS, ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyspectrum::calc_RegPT_1loop(np::ndarray k){
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for(int i=0;i<N;++i){
    ki = p::extract<double>(k[i]);
    Pi = Spectra->Preg_1loop(DENS, DENS, ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

np::ndarray pyspectrum::calc_SPT_1loop(np::ndarray k){
  double ki, Pi;
  size_t N;
  vector<double> res;

  check_array(k);
  N = k.shape(0);

  for(int i=0;i<N;++i){
    ki = p::extract<double>(k[i]);
    Pi = Spectra->Pspt_1loop(DENS, DENS, ki);
    res.push_back(Pi);
  }

  return vec_to_ndarray(res);
}

BOOST_PYTHON_MODULE(pyeclairs){
  np::initialize();

  p::class_<pyspectrum>("pyspectrum")
      .def("set_cosmology", &pyspectrum::set_cosmology)
      .def("calc_linear", &pyspectrum::calc_linear)
      .def("calc_no_wiggle", &pyspectrum::calc_no_wiggle)
      .def("calc_RegPT_2loop", &pyspectrum::calc_RegPT_2loop)
      .def("calc_SPT_2loop", &pyspectrum::calc_SPT_2loop)
      .def("calc_RegPT_1loop", &pyspectrum::calc_RegPT_1loop)
      .def("calc_SPT_1loop", &pyspectrum::calc_SPT_1loop)
  ;

}
