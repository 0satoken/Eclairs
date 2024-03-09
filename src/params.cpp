#include "params.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

params::params(void) {
  initialize();
  set_default_parameter();
  check_conflict();
  if (bparams["verbose"])
    show_parameter();
}

params::params(char *params_fname) {
  initialize();
  set_default_parameter();
  load_parameter(params_fname);
  check_conflict();
  if (bparams["verbose"])
    show_parameter();
}

void params::initialize(void) {
  name_list = {"H0",
               "Omega_m",
               "Omega_b",
               "Omega_k",
               "ns",
               "As",
               "w_de",
               "k_pivot",
               "m_nu",
               "z",
               "transfer_EH",
               "transfer_from_file",
               "transfer_file_name",
               "lambda_power",
               "lambda_bispectrum",
               "free_sigma_d",
               "sigma_d",
               "b1",
               "b2",
               "bs2",
               "b3nl",
               "N_shot",
               "sigma_v",
               "use_sigma_vlin",
               "alpha_perp",
               "alpha_para",
               "rs_drag_ratio",
               "AP",
               "FoG_type",
               "gamma",
               "multipole_nmu",
               "grid_L",
               "grid_ng",
               "kmin",
               "kmax",
               "nint",
               "nk_G1_1",
               "nk_G1_2",
               "nk_G2_1",
               "nx_Pcorr2",
               "nmu_Pcorr2",
               "MC_calls",
               "MC_tol",
               "output",
               "output_model",
               "output_1loop",
               "output_file_name",
               "output_spacing",
               "output_kmin",
               "output_kmax",
               "output_nk",
               "direct_mode",
               "nk_spl",
               "direct_nr",
               "direct_nx",
               "direct_nq",
               "direct_nmu",
               "direct_nphi",
               "direct_mumin",
               "direct_mumax",
               "direct_phimin",
               "direct_phimax",
               "direct_MC",
               "direct_MC_calls",
               "direct_MC_tol",
               "direct_SPT",
               "direct_1loop",
               "direct_spline",
               "fast_mode",
               "fast_qmin",
               "fast_qmax",
               "fast_nq",
               "fast_mumin",
               "fast_mumax",
               "fast_nmu",
               "fast_phimin",
               "fast_phimax",
               "fast_nphi",
               "fast_nr",
               "fast_nx",
               "fast_nkb",
               "fast_direct_Bterm",
               "fast_SPT",
               "fast_fidmodels_dir",
               "fast_fidmodels_config",
               "fast_fidmodels_k1min",
               "fast_fidmodels_k1max",
               "fast_fidmodels_nk1",
               "fast_fidmodels_k2min",
               "fast_fidmodels_k2max",
               "fast_fidmodels_nk2",
               "kernel_root",
               "kernel_k_from_file",
               "kernel_k_file_name",
               "kernel_k_spacing",
               "kernel_k_kmin",
               "kernel_k_kmax",
               "kernel_k_nk",
               "kernel_K_Kmin",
               "kernel_K_Kmax",
               "kernel_K_nK",
               "smooth_nk",
               "smooth_kmin",
               "smooth_kmax",
               "smooth_lambda",
               "IREFT_mode",
               "IREFT_c0",
               "IREFT_c2",
               "IREFT_c4",
               "IREFT_cd4",
               "IREFT_Pshot",
               "IREFT_kS",
               "IREFT_kM",
               "IREFT_rs",
               "IREFT_nr",
               "IREFT_nx",
               "IREFT_nq",
               "IREFT_nmu",
               "bias_higher_order",
               "bias_local_Lagrangian_bias",
               "bias_spline",
               "bias_nr",
               "bias_nx",
               "verbose"};

  type_list = {
      {"H0", DOUBLE},
      {"Omega_m", DOUBLE},
      {"Omega_b", DOUBLE},
      {"Omega_k", DOUBLE},
      {"ns", DOUBLE},
      {"As", DOUBLE},
      {"w_de", DOUBLE},
      {"k_pivot", DOUBLE},
      {"m_nu", DOUBLE},
      {"z", DOUBLE},
      {"transfer_EH", BOOL},
      {"transfer_from_file", BOOL},
      {"transfer_file_name", STRING},
      {"lambda_power", DOUBLE},
      {"lambda_bispectrum", DOUBLE},
      {"free_sigma_d", BOOL},
      {"sigma_d", DOUBLE},
      {"b1", DOUBLE},
      {"b2", DOUBLE},
      {"bs2", DOUBLE},
      {"b3nl", DOUBLE},
      {"N_shot", DOUBLE},
      {"sigma_v", DOUBLE},
      {"use_sigma_vlin", BOOL},
      {"alpha_perp", DOUBLE},
      {"alpha_para", DOUBLE},
      {"rs_drag_ratio", DOUBLE},
      {"AP", BOOL},
      {"FoG_type", STRING},
      {"gamma", DOUBLE},
      {"multipole_nmu", INT},
      {"grid_L", DOUBLE},
      {"grid_ng", INT},
      {"kmin", DOUBLE},
      {"kmax", DOUBLE},
      {"nint", INT},
      {"nk_G1_1", INT},
      {"nk_G1_2", INT},
      {"nk_G2_1", INT},
      {"nx_Pcorr2", INT},
      {"nmu_Pcorr2", INT},
      {"MC_calls", INT},
      {"MC_tol", DOUBLE},
      {"output", BOOL},
      {"output_model", STRING},
      {"output_1loop", BOOL},
      {"output_file_name", STRING},
      {"output_spacing", STRING},
      {"output_kmin", DOUBLE},
      {"output_kmax", DOUBLE},
      {"direct_mode", BOOL},
      {"nk_spl", INT},
      {"output_nk", INT},
      {"direct_nr", INT},
      {"direct_nx", INT},
      {"direct_nq", INT},
      {"direct_nmu", INT},
      {"direct_nphi", INT},
      {"direct_mumin", DOUBLE},
      {"direct_mumax", DOUBLE},
      {"direct_phimin", DOUBLE},
      {"direct_phimax", DOUBLE},
      {"direct_MC", BOOL},
      {"direct_MC_calls", INT},
      {"direct_MC_tol", DOUBLE},
      {"direct_SPT", BOOL},
      {"direct_1loop", BOOL},
      {"direct_spline", BOOL},
      {"fast_mode", BOOL},
      {"fast_qmin", DOUBLE},
      {"fast_qmax", DOUBLE},
      {"fast_nq", INT},
      {"fast_mumin", DOUBLE},
      {"fast_mumax", DOUBLE},
      {"fast_nmu", INT},
      {"fast_phimin", DOUBLE},
      {"fast_phimax", DOUBLE},
      {"fast_nphi", INT},
      {"fast_nr", INT},
      {"fast_nx", INT},
      {"fast_nkb", INT},
      {"fast_direct_Bterm", BOOL},
      {"fast_SPT", BOOL},
      {"fast_fidmodels_dir", STRING},
      {"fast_fidmodels_config", STRING},
      {"fast_fidmodels_k1min", DOUBLE},
      {"fast_fidmodels_k1max", DOUBLE},
      {"fast_fidmodels_nk1", INT},
      {"fast_fidmodels_k2min", DOUBLE},
      {"fast_fidmodels_k2max", DOUBLE},
      {"fast_fidmodels_nk2", INT},
      {"kernel_root", STRING},
      {"kernel_k_from_file", BOOL},
      {"kernel_k_file_name", STRING},
      {"kernel_k_spacing", STRING},
      {"kernel_k_kmin", DOUBLE},
      {"kernel_k_kmax", DOUBLE},
      {"kernel_k_nk", INT},
      {"kernel_K_Kmin", DOUBLE},
      {"kernel_K_Kmax", DOUBLE},
      {"kernel_K_nK", INT},
      {"smooth_nk", INT},
      {"smooth_kmin", DOUBLE},
      {"smooth_kmax", DOUBLE},
      {"smooth_lambda", DOUBLE},
      {"IREFT_mode", BOOL},
      {"IREFT_c0", DOUBLE},
      {"IREFT_c2", DOUBLE},
      {"IREFT_c4", DOUBLE},
      {"IREFT_cd4", DOUBLE},
      {"IREFT_Pshot", DOUBLE},
      {"IREFT_kS", DOUBLE},
      {"IREFT_kM", DOUBLE},
      {"IREFT_rs", DOUBLE},
      {"IREFT_nr", INT},
      {"IREFT_nx", INT},
      {"IREFT_nq", INT},
      {"IREFT_nmu", INT},
      {"bias_higher_order", BOOL},
      {"bias_local_Lagrangian_bias", BOOL},
      {"bias_spline", BOOL},
      {"bias_nr", INT},
      {"bias_nx", INT},
      {"verbose", BOOL},
  };

  return;
}

void params::set_default_parameter(void) {
  double h, pi;

  pi = 4.0 * atan(1.0);

  /* cosmological parameters */
  dparams["H0"] = 70.10;
  h = dparams["H0"] / 100.0;
  dparams["Omega_m"] = (0.114479234 + 0.022621645) / (h * h);
  dparams["Omega_b"] = 0.022621645 / (h * h);
  dparams["Omega_k"] = 0.0;
  dparams["ns"] = 0.96;
  dparams["As"] = 2.1777e-9;
  dparams["w_de"] = -1.0;
  dparams["k_pivot"] = 0.05; //[Mpc^-1]
  dparams["m_nu"] = 0.06;    // [eV]
  dparams["z"] = 0.0;

  /* transfer function setting */
  bparams["transfer_EH"] = true; // use Eisenstein & Hu transfer function
  bparams["transfer_from_file"] = false; // load transfer function from table
  sparams["transfer_file_name"] = "";    // file name of the tabulated data

  /* UV cutoff parameter for sigmad */
  dparams["lambda_power"] = 2.0;      // UV cutoff parameter for power spectrum
  dparams["lambda_bispectrum"] = 6.0; // UV cutoff parameter for bispectrum

  /* free sigma_d model (RegPT+) setting */
  bparams["free_sigma_d"] = false; // switch to RegPT+
  dparams["sigma_d"] = 3.0;        // damping factor in [Mpc/h]

  /* parameters relavant to redshift space distortion */
  dparams["b1"] = 1.0;      // linear galaxy bias (b1 = 1 corresponds to matter)
  dparams["b2"] = 0.0;
  dparams["bs2"] = 0.0;
  dparams["b3nl"] = 0.0;
  dparams["N_shot"] = 0.0;
  dparams["sigma_v"] = 0.0; // velocity dispersion relevant for FoG effect
  bparams["use_sigma_vlin"] = false; // velocity dispersion computed at linear order
  dparams["alpha_perp"] = 1.0;    // AP perpendicular parameter
  dparams["alpha_para"] = 1.0;    // AP parallel parameter
  dparams["rs_drag_ratio"] = 1.0; // ratio of rs_drag (= rs_drag/rs_drag_fid)
  bparams["AP"] = false;          // flag for Alcock-Paczynski effect
  sparams["FoG_type"] = "Lorentzian"; // functional form of FoG damping factor ("Gaussian",
                          // "Lorentzian", "Gamma", "None")
  dparams["gamma"] = 2.0; // slope factor for Gamma FoG model
  iparams["multipole_nmu"] = 50; // integration step for mu in multipole decomposition
  dparams["grid_L"] = 2048.0; // box size used in finite grid summation
  iparams["grid_ng"] = 256;   // grid size used in finite grid summation

  /* parameters relavant to integration */
  dparams["kmin"] = 5e-4; // minimum wavenumber in integration [h/Mpc]
  dparams["kmax"] = 10.0; // maximum wavenumber in integration [h/Mpc]
  iparams["nint"] = 500;  // steps for generic integration in cosmology class

  /* Precision parameters for power spectrum calculation */
  iparams["nk_G1_1"] = 2000;    // integration step for Gamma1_1loop
  iparams["nk_G1_2"] = 500;     // integration step for Gamma1_2loop
  iparams["nk_G2_1"] = 100;     // integration step for Gamma2_1loop
  iparams["nx_Pcorr2"] = 200;   // integration step in radial direction
  iparams["nmu_Pcorr2"] = 10;   // integration step in azimuthal direction
  iparams["MC_calls"] = 500000; // Monte-Carlo step
  dparams["MC_tol"] = 0.5;      // tolerance of Monte-Carlo integration

  /* output settings */
  bparams["output"] = true;                   // flag for output
  sparams["output_model"] = "RegPT";          // RegPT or SPT
  bparams["output_1loop"] = false;            // switch to 1-loop level
  sparams["output_file_name"] = "output.dat"; // output file name
  sparams["output_spacing"] = "linear";       // spacing of output wavenumbers
  dparams["output_kmin"] = 0.01;              // minimum wavenumber for output
  dparams["output_kmax"] = 0.5;               // maximum wavenumber for output
  iparams["output_nk"] = 10;                  // number of wavenumber bins

  /* parameters for redshift space spectra in direct mode */
  bparams["direct_mode"] = true; // flag for direct mode (default)
  iparams["nk_spl"] = 1000;   // number of wavenumbers for sigmad/1-loop spline
  iparams["direct_nr"] = 600; // steps for r in integration
  iparams["direct_nx"] = 10;  // steps for x in integration
  iparams["direct_nq"] = 400; // steps for q in integration
  dparams["direct_mumin"] = -0.999;              // minimum mu in integration
  dparams["direct_mumax"] = 0.999;               // maximum mu in integration
  iparams["direct_nmu"] = 50;                    // steps for mu in integration
  dparams["direct_phimin"] = 0.001 * (2.0 * pi); // minimum phi in integration
  dparams["direct_phimax"] = 0.999 * (2.0 * pi); // maximum phi in integration
  iparams["direct_nphi"] = 50;                   // steps for phi in integration
  bparams["direct_MC"] = true; // if true, Monte-Carlo integration used for A-term
  iparams["direct_MC_calls"] = 500000; // Monte-Carlo step
  dparams["direct_MC_tol"] = 0.5;      // tolerance of Monte-Carlo integration
  bparams["direct_SPT"] = false;       // if true, SPT employed
  bparams["direct_1loop"] = false;     // if true, 1loop order
  bparams["direct_spline"] = false;    // if true, precompute power spectrum and
                                       // TNS terms, and use spline for evaluation

  /* parameters for fast reconstruction */
  bparams["fast_mode"] = false;   // flag for fast mode
  dparams["fast_qmin"] = 5e-4;    // minimum wavenumber in integration [h/Mpc]
  dparams["fast_qmax"] = 10.0;    // maximum wavenumber in integration [h/Mpc]
  iparams["fast_nq"] = 200;       // steps for q in integration
  dparams["fast_mumin"] = -0.999; // minimum mu in integration
  dparams["fast_mumax"] = 0.999;  // maximum mu in integration
  iparams["fast_nmu"] = 50;       // steps for mu in integration
  dparams["fast_phimin"] = 0.001 * (2.0 * pi); // minimum phi in integration
  dparams["fast_phimax"] = 0.999 * (2.0 * pi); // maximum phi in integration
  iparams["fast_nphi"] = 50;                   // steps for phi in integration
  iparams["fast_nr"] = 600;                    // steps for r in integration
  iparams["fast_nx"] = 10;                     // steps for x in integration
  iparams["fast_nkb"] = 50; // steps for bispectrum binning summation
  bparams["fast_direct_Bterm"] = false; // if true, for B-term 1-loop power is calculated with direct mode
  bparams["fast_SPT"] = false; // if true, SPT employed
  sparams["fast_fidmodels_dir"] = "kernel_data"; // directory of kernels for fiducial models
  sparams["fast_fidmodels_config"] = "config.dat"; // configuration file for fiducial models
  dparams["fast_fidmodels_k1min"] = 0.01;
  dparams["fast_fidmodels_k1max"] = 1.0;
  iparams["fast_fidmodels_nk1"] = 20;
  dparams["fast_fidmodels_k2min"] = 0.15;
  dparams["fast_fidmodels_k2max"] = 1.0;
  iparams["fast_fidmodels_nk2"] = 20;
  sparams["kernel_root"] = "kernel_data"; // output for kernel data
  bparams["kernel_k_from_file"] = false; // flag to load wavenumber bins from file
  sparams["kernel_k_file_name"] = "kernel_k.dat"; // output for kernel data
  sparams["kernel_k_spacing"] = "log"; // spacing of output wavenumbers
  dparams["kernel_k_kmin"] = 1e-3;     // minimum wavenumber for output
  dparams["kernel_k_kmax"] = 1.0;      // maximum wavenumber for output
  iparams["kernel_k_nk"] = 120;        // number of wavenumber bins
  dparams["kernel_K_Kmin"] = 1e-3; // minimum wavenumber for output for fast-bispectrum
  dparams["kernel_K_Kmax"] = 0.6; // maximum wavenumber for output for fast-bispectrum
  iparams["kernel_K_nK"] = 100; // number of wavenumber bins for fast-bispectrum

  /* parameters for IR-resummed EFT */
  bparams["IREFT_mode"] = false;   // flag for IR EFT mode
  iparams["smooth_nk"] = 500;      // number of steps for smoothing
  dparams["smooth_kmin"] = 1e-6;   // minimum k for smoothing
  dparams["smooth_kmax"] = 50.0;   // maximum k for smoothing
  dparams["smooth_lambda"] = 0.25; // smoothing length
  dparams["IREFT_c0"] = 0.0;       // EFT parameter c0
  dparams["IREFT_c2"] = 0.0;       // EFT parameter c2
  dparams["IREFT_c4"] = 0.0;       // EFT parameter c4
  dparams["IREFT_cd4"] = 0.0;      // EFT parameter cd4
  dparams["IREFT_Pshot"] = 0.0;    // shot noise level [(Mpc/h)^3]
  dparams["IREFT_kS"] = 0.2;       // cutoff for Sigma integration [h/Mpc]
  dparams["IREFT_rs"] = 110.0;     // sound horizon at decoupling [Mpc/h]
  iparams["IREFT_nr"] = 600;
  iparams["IREFT_nx"] = 50;
  iparams["IREFT_nq"] = 200;
  iparams["IREFT_nmu"] = 50;

  /* parameters for galaxy bias */
  bparams["bias_higher_order"] = false; // flag for higher order bias
  bparams["bias_local_Lagrangian_bias"] = false; // flag for local Lagrangian bias
  bparams["bias_spline"] = true;                // flag for spline
  iparams["bias_nr"] = 600;                      // steps for r in integration
  iparams["bias_nx"] = 50;                       // steps for x in integration

  /* miscellanea */
  bparams["verbose"] = false;

  return;
}

void params::load_parameter(const char *ini_fname) {
  ifstream ifs;
  string str;

  ifs.open(ini_fname, ios::in);
  if (ifs.fail()) {
    cerr << "[ERROR] parameter file open error!:" << ini_fname << endl;
    exit(1);
  }

  while (getline(ifs, str)) {
    /* remove spaces */
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    /* erase comments */
    if (str.find_first_of('#') != string::npos)
      str.erase(str.find_first_of('#'));
    /* ignore empty lines */
    if (str == "")
      continue;

    stringstream ss(str);
    string s;
    vector<string> v;

    while (getline(ss, s, '=')) {
      v.push_back(s);
    }

    if (v.size() != 2) {
      cerr << "[ERROR] Invalid line!: " << str << endl;
      exit(1);
    }

    store_parameter(v[0], v[1]);
  }

  return;
}

void params::store_parameter(string pname, string p) {
  if (type_list.count(pname) == 0) {
    cerr << "[ERROR] The parameter \"" << pname << "\" is not allowed." << endl;
    exit(1);
  }

  switch (type_list[pname]) {
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
    if (p == "True" || p == "true")
      bparams[pname] = true;
    else if (p == "False" || p == "false")
      bparams[pname] = false;
    else {
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

void params::show_parameter(void) {
  string pname;

  cout << "-> showing parameters" << endl;
  for (size_t i = 0; i < name_list.size(); ++i) {
    pname = name_list[i];
    switch (type_list[pname]) {
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
      if (bparams[pname])
        cout << pname << " = True" << endl;
      else
        cout << pname << " = False" << endl;
      break;
    default:
      break;
    }
  }

  cout << endl;
  return;
}

void params::check_conflict(void) {
  /*
  if(!bparams["transfer_EH"] != !bparams["transfer_from_file"]){
    cerr << "Either of \"transfer_EH\" or \"transfer_from_file\" must be True."
  << endl; exit(1);
  }
  */

  return;
}

vector<string> params::get_name_list(void) { return name_list; }

map<string, TYPE> params::get_type_list(void) { return type_list; }
