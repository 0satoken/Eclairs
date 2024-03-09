#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include "io.hpp"


params::params(void){
  initialize();
  set_default_parameter();
  check_conflict();
  if(bparams["verbose"]) show_parameter();
}

params::params(char *params_fname){
  initialize();
  set_default_parameter();
  load_parameter(params_fname);
  check_conflict();
  if(bparams["verbose"]) show_parameter();
}

void params::initialize(void){
  name_list = { "H0", "Omega_m", "Omega_b", "ns", "As", "k_pivot", "m_nu", "z",
                "transfer_EH", "transfer_from_file", "transfer_file_name",
                "free_sigma_d", "sigma_d", "smoothed_linear_power",
                "kmin", "kmax", "nint", "nk_G1_1", "nk_G1_2", "nk_G2_1",
                "nx", "nmu", "MC_calls", "MC_tol",
                "output", "output_model", "output_1loop",
                "output_file_name", "output_spacing",
                "output_kmin", "output_kmax", "output_nk",
                "verbose"
              };

  type_list = {
    {"H0", DOUBLE},
    {"Omega_m", DOUBLE},
    {"Omega_b", DOUBLE},
    {"ns", DOUBLE},
    {"As", DOUBLE},
    {"k_pivot", DOUBLE},
    {"m_nu", DOUBLE},
    {"z", DOUBLE},
    {"transfer_EH", BOOL},
    {"transfer_from_file", BOOL},
    {"transfer_file_name", STRING},
    {"free_sigma_d", BOOL},
    {"sigma_d", DOUBLE},
    {"smoothed_linear_power", BOOL},
    {"kmin", DOUBLE},
    {"kmax", DOUBLE},
    {"nint", INT},
    {"nk_G1_1", INT},
    {"nk_G1_2", INT},
    {"nk_G2_1", INT},
    {"nx", INT},
    {"nmu", INT},
    {"MC_calls", INT},
    {"MC_tol", DOUBLE},
    {"output", BOOL},
    {"output_model", STRING},
    {"output_1loop", BOOL},
    {"output_file_name", STRING},
    {"output_spacing", STRING},
    {"output_kmin", DOUBLE},
    {"output_kmax", DOUBLE},
    {"output_nk", INT},
    {"verbose", BOOL},
  };

  return;
}

void params::set_default_parameter(void){
  double h;

  /* cosmological parameters */
  dparams["H0"] = 70.10;
  h = dparams["H0"]/100.0;
  dparams["Omega_m"] = (0.114479234+0.022621645)/(h*h);
  dparams["Omega_b"] = 0.022621645/(h*h);
  dparams["ns"] = 0.96;
  dparams["As"] = 2.1777e-9;
  dparams["k_pivot"] = 0.05; //[Mpc^-1]
  dparams["m_nu"] = 0.06; // [eV]
  dparams["z"] = 0.0;

  /* transfer function setting */
  bparams["transfer_EH"] = true; // use Eisenstein & Hu transfer function
  bparams["transfer_from_file"] = false; // load transfer function from table
  sparams["transfer_file_name"] = ""; // name of the tabulated data

  /* free sigma_d model (RegPT+) setting */
  bparams["free_sigma_d"] = false; // switch to RegPT+
  dparams["sigma_d"] = 3.0; // damping factor in [Mpc/h]

  /* smoothed linear power spectrum */
  bparams["smoothed_linear_power"] = false;

  /* parameters related with integration */
  dparams["kmin"] = 5e-4; // minimum wavenumber in integration
  dparams["kmax"] = 10.0; // maximum wavenumber in integration
  iparams["nint"] = 1000; // steps for generic integration in cosmology class

  /* Precision parameters for main calculation */
  iparams["nk_G1_1"] = 2000; // integration step for Gamma1_1loop
  iparams["nk_G1_2"] = 500; // integration step for Gamma1_2loop
  iparams["nk_G2_1"] = 100; // integration step for Gamma2_1loop
  iparams["nx"] = 200; // integration step in radial direction
  iparams["nmu"] = 10; // integration step in azimuthal direction
  iparams["MC_calls"] = 500000; // Monte-Carlo step
  dparams["MC_tol"] = 0.5; // tolerance of Monte-Carlo integration

  /* output settings */
  bparams["output"] = true;
  sparams["output_model"] = "RegPT";
  bparams["output_1loop"] = false;
  sparams["output_file_name"] = "output.dat";
  sparams["output_spacing"] = "linear";
  dparams["output_kmin"] = 0.01;
  dparams["output_kmax"] = 0.5;
  iparams["output_nk"] = 10;

  /* miscellanea */
  bparams["verbose"] = false;

  return;
}

void params::load_parameter(const char *ini_fname){
  ifstream ifs;
  string str;

  ifs.open(ini_fname, ios::in);
  if(ifs.fail()){
    cerr << "[ERROR] parameter file open error!:" << ini_fname << endl;
    exit(1);
  }

  while(getline(ifs, str)){
    /* remove spaces */
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    /* erase comments */
    if(str.find_first_of('#') != string::npos) str.erase(str.find_first_of('#'));
    /* ignore empty lines */
    if(str == "") continue;

    stringstream ss(str);
    string s;
    vector<string> v;

    while(getline(ss, s, '=')){
      v.push_back(s);
    }

    if(v.size() != 2){
      cerr << "[ERROR] Invalid line!: "<< str << endl;
      exit(1);
    }

    store_parameter(v[0], v[1]);
  }

  return;
}

void params::store_parameter(string pname, string p){
  if(type_list.count(pname) == 0){
    cerr << "[ERROR] The parameter \"" << pname << "\" is not allowed." << endl;
    exit(1);
  }

  switch(type_list[pname]){
    case INT:
      iparams[pname] = stoi(p);
      break;
    case DOUBLE:
      dparams[pname] = stod(p);
      break;
    case STRING:
      sparams[pname] = p;
      break;
    case BOOL:
      if(p == "True" || p == "true") bparams[pname] = true;
      else if(p == "False" || p == "false") bparams[pname] = false;
      else{
        cerr << "[ERROR] The parameter \"" << pname << " = " << p;
        cerr << "\" is invalid for boolean" << endl;
        exit(1);
      }
      break;
    default:
      break;
  }

  return;
}

void params::show_parameter(void){
  string pname;

  cout << "-> showing parameters" << endl;
  for(size_t i=0;i<name_list.size();++i){
    pname = name_list[i];
    switch(type_list[pname]){
      case INT:
        cout << pname << " = " << iparams[pname] << endl;
        break;
      case DOUBLE:
        cout << pname << " = " << dparams[pname] << endl;
        break;
      case STRING:
        cout << pname << " = " << sparams[pname] << endl;
        break;
      case BOOL:
        if(bparams[pname]) cout << pname << " = True" << endl;
        else cout << pname << " = False" << endl;
        break;
      default:
        break;
    }
  }

  cout << endl;
  return;
}

void params::check_conflict(void){
  /*
  if(!bparams["transfer_EH"] != !bparams["transfer_from_file"]){
    cerr << "Either of \"transfer_EH\" or \"transfer_from_file\" must be True." << endl;
    exit(1);
  }
  */

  return;
}
