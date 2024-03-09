#include "cosmology.hpp"
#include "direct_red.hpp"
#include "fast_bispectra.hpp"
#include "fast_spectra.hpp"
#include "kernel.hpp"
#include "nonlinear.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include "spectra_red.hpp"
#include "vector.hpp"
#include <fstream>
#include <iostream>
#include <string>

#include <chrono>

#ifdef MPI_PARALLEL
#include "validation.hpp"
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

using namespace std;

void validate_Pk_minimal(char *inifile, char *fname, mpi::communicator &world) {
  int myrank, numprocs;
  const int nk = 200;
  const double kmin = 1e-3;
  const double kmax = 0.3;
  double ki;
  double k[nk];
  double *Pk, *Pk0;
  FILE *fp;

  myrank = world.rank();
  numprocs = world.size();

  params Params(inifile);
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);

  for (int i = 0; i < nk; ++i) {
    k[i] = (kmax - kmin) / (nk - 1.0) * (i + 0.5);
  }

  Pk = new double[3 * nk];
  Pk0 = new double[3 * nk];

  for (int i = 0; i < 3 * nk; ++i) {
    Pk[i] = 0.0;
    Pk0[i] = 0.0;
  }

  for (int i = myrank; i < nk; i += numprocs) {
    ki = k[i];

    Pk[0 + 3 * i] = Spectra.Preg_2loop(DENS, DENS, ki);
    Pk[1 + 3 * i] = Spectra.Preg_2loop(DENS, VELO, ki);
    Pk[2 + 3 * i] = Spectra.Preg_2loop(VELO, VELO, ki);
  }

  all_reduce(world, Pk, 3 * nk, Pk0, plus<double>());

  if (myrank == 0) {
    if ((fp = fopen(fname, "w")) == NULL) {
      cerr << "file open error" << endl;
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e", k[i], Cosmo.Plin(k[i]), Cosmo.Pno_wiggle(k[i]));
      for (int j = 0; j < 3; ++j)
        fprintf(fp, " %e", Pk0[j + 3 * i]);
      fprintf(fp, "\n");
    }

    fclose(fp);
  }

  delete[] Pk;
  delete[] Pk0;

  return;
}

void validate_Pk_TNS(char *inifile, char *fname, mpi::communicator &world) {
  int myrank, numprocs;
  const int nk = 200;
  const double kmin = 1e-3;
  const double kmax = 0.3;
  double ki;
  double k[nk];
  double *Pk, *Pk0;
  map<string, double> res_A, res_B;
  FILE *fp;

  myrank = world.rank();
  numprocs = world.size();

  params Params(inifile);
  Params.bparams["direct_spline"] = false;
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);
  direct_red Direct_red(Params, Cosmo, Spectra);

  for (int i = 0; i < nk; ++i) {
    k[i] = (kmax - kmin) / (nk - 1.0) * (i + 0.5);
  }

  Pk = new double[10 * nk];
  Pk0 = new double[10 * nk];

  for (int i = 0; i < 10 * nk; ++i) {
    Pk[i] = 0.0;
    Pk0[i] = 0.0;
  }

  for (int i = myrank; i < nk; i += numprocs) {
    ki = k[i];

    Pk[0 + 10 * i] = Spectra.Preg_2loop(DENS, DENS, ki);
    Pk[1 + 10 * i] = Spectra.Preg_2loop(DENS, VELO, ki);
    Pk[2 + 10 * i] = Spectra.Preg_2loop(VELO, VELO, ki);

    res_A = Direct_red.get_Aterm(ki);

    Pk[3 + 10 * i] = res_A["A2"];
    Pk[4 + 10 * i] = res_A["A4"];
    Pk[5 + 10 * i] = res_A["A6"];

    res_B = Direct_red.get_Bterm(ki);

    Pk[6 + 10 * i] = res_B["B2"];
    Pk[7 + 10 * i] = res_B["B4"];
    Pk[8 + 10 * i] = res_B["B6"];
    Pk[9 + 10 * i] = res_B["B8"];
  }

  all_reduce(world, Pk, 10 * nk, Pk0, plus<double>());

  if (myrank == 0) {
    if ((fp = fopen(fname, "w")) == NULL) {
      cerr << "file open error" << endl;
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e", k[i], Cosmo.Plin(k[i]), Cosmo.Pno_wiggle(k[i]));
      for (int j = 0; j < 10; ++j)
        fprintf(fp, " %e", Pk0[j + 10 * i]);
      fprintf(fp, "\n");
    }

    fclose(fp);
  }

  delete[] Pk;
  delete[] Pk0;

  return;
}

void validate_Pk_red(char *inifile, char *fname, mpi::communicator &world) {
  int myrank, numprocs;
  const int nk = 120;
  const double kmin = 1e-3;
  const double kmax = 0.3;
  double ki;
  double k[nk];
  double *Pk, *Pk0;
  vector<double> kvec, w3, w2;
  vector<int> l;
  vector<vector<double>> res;
  FILE *fp;

  myrank = world.rank();
  numprocs = world.size();

  params Params(inifile);
  Params.bparams["direct_spline"] = false;
  Params.bparams["use_sigma_vlin"] = true;
  Params.dparams["lambda_power"] = 2.0;
  Params.dparams["lambda_bispectrum"] = 2.0;
  Params.dparams["z"] = 1.0;

  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);

  direct_red Direct_red(Params, Cosmo, Spectra);
  spectra_red Spectra_red(Params, Cosmo, Direct_red);

  //fast_spectra Fast_spectra(Params, Cosmo, Spectra);
  //spectra_red Spectra_red(Params, Cosmo, Fast_spectra);

  for (int i = 0; i < nk; ++i) {
    k[i] = (kmax - kmin) / (nk - 1.0) * (i + 0.5);
  }

  kvec.resize(1);
  l.resize(3);
  w3.resize(4);
  w2.resize(3);

  l[0] = 0;
  l[1] = 2;
  l[2] = 4;

  w3[0] = 0.0;
  w3[1] = 1.0 / 3.0;
  w3[2] = 2.0 / 3.0;
  w3[3] = 1.0;

  w2[0] = 0.0;
  w2[1] = 1.0 / 2.0;
  w2[2] = 1.0;

  Pk = new double[8 * nk];
  Pk0 = new double[8 * nk];

  for (int i = 0; i < 8 * nk; ++i) {
    Pk[i] = 0.0;
    Pk0[i] = 0.0;
  }


  for (int i = myrank; i < nk; i += numprocs) {
    kvec[0] = k[i];

    res = Spectra_red.get_multipoles(kvec, l);
    printf("# k:%g -> multipole done\n", k[i]);

    Pk[0 + 8 * i] = res[0][0];
    Pk[1 + 8 * i] = res[0][1];
    Pk[2 + 8 * i] = res[0][2];

    res = Spectra_red.get_wedges(kvec, w3);
    printf("# k:%g -> 3-wedges done\n", k[i]);

    Pk[3 + 8 * i] = res[0][0];
    Pk[4 + 8 * i] = res[0][1];
    Pk[5 + 8 * i] = res[0][2];

    res = Spectra_red.get_wedges(kvec, w2);
    printf("# k:%g -> 2-wedges done\n", k[i]);

    Pk[6 + 8 * i] = res[0][0];
    Pk[7 + 8 * i] = res[0][1];
  }

  all_reduce(world, Pk, 8 * nk, Pk0, plus<double>());

  if (myrank == 0) {
    if ((fp = fopen(fname, "w")) == NULL) {
      cerr << "file open error" << endl;
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e", k[i], Cosmo.Plin(k[i]), Cosmo.Pno_wiggle(k[i]));
      for (int j = 0; j < 8; ++j)
        fprintf(fp, " %e", Pk0[j + 8 * i]);
      fprintf(fp, "\n");
    }

    fclose(fp);
  }

  delete[] Pk;
  delete[] Pk0;

  return;
}

void validate_Bk(char *inifile, char *fname, mpi::communicator &world) {
  int myrank, numprocs;
  const int nk = 500;
  const double kmin = 1e-3;
  const double kmax = 0.3;
  double k[nk];
  double k1, k2, k3, ki;
  double *Bk, *Bk0;
  FILE *fp;
  double kiso1[] = {0.015, 0.035, 0.055, 0.075, 0.095};
  double kiso2[] = {0.075, 0.095, 0.115, 0.135, 0.155};

  myrank = world.rank();
  numprocs = world.size();

  params Params(inifile);
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);
  bispectra Bispectra(Params, Cosmo, Spectra);

  for (int i = 0; i < nk; ++i) {
    k[i] = (kmax - kmin) / (nk - 1.0) * (i + 0.5);
  }

  Bk = new double[11 * nk];
  Bk0 = new double[11 * nk];

  for (int i = 0; i < 11 * nk; ++i) {
    Bk[i] = 0.0;
    Bk0[i] = 0.0;
  }

  for (int i1 = 0; i1 < nk; ++i1) {
    ki = k[i1];
    for (int i2 = 0; i2 < 11; ++i2) {
      if ((i2 + 11 * i1) % numprocs != myrank)
        continue;

      if (i2 < 1) {
        k1 = ki;
        k2 = ki;
        k3 = ki;
      } else if (1 <= i2 && i2 < 6) {
        k1 = ki;
        k2 = ki;
        k3 = kiso1[i2 - 1];
      } else {
        k1 = ki;
        k2 = kiso2[i2 - 6];
        k3 = kiso2[i2 - 6];
      }

      if (k1 + k2 < k3 || k2 + k3 < k1 || k3 + k1 < k2) {
        Bk[i2 + 11 * i1] = 0.0;
      } else {
        Bk[i2 + 11 * i1] = Bispectra.Bispec_1loop(DENS, DENS, DENS, k1, k2, k3);
      }
    }
  }

  all_reduce(world, Bk, 11 * nk, Bk0, plus<double>());

  if (myrank == 0) {
    if ((fp = fopen(fname, "w")) == NULL) {
      cerr << "file open error" << endl;
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e", k[i]);
      for (int j = 0; j < 11; ++j)
        fprintf(fp, " %e", Bk0[j + 11 * i]);
      fprintf(fp, "\n");
    }

    fclose(fp);
  }

  delete[] Bk;
  delete[] Bk0;

  return;
}

void validate_recon_Pk(char *inifile, char *header, mpi::communicator &world) {
  int myrank, numprocs;
  char fname[256];
  double ki;
  map<string, double> res_Pk, res_A, res_B;
  FILE *fp;

  const int nk = 120;
  double k[nk];
  const double kmin = 1e-3;
  const double kmax = 0.3;
  double *Pk_tar, *Pk_tar0, *Pk_recon, *Pk_recon0;

  myrank = world.rank();
  numprocs = world.size();

  params Params(inifile);
  Params.bparams["direct_spline"] = false;
  Params.dparams["lambda_power"] = 2.0;
  Params.dparams["lambda_bispectrum"] = 2.0;
  Params.dparams["z"] = 1.0;

  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);
  direct_red Direct_red(Params, Cosmo, Spectra);
  fast_spectra Fast_spectra(Params, Cosmo, Spectra);

  for (int i = 0; i < nk; ++i) {
    k[i] = (kmax - kmin) / (nk - 1.0) * (i + 0.5);
  }

  Pk_tar = new double[nk * 10];
  Pk_tar0 = new double[nk * 10];
  Pk_recon = new double[nk * 10];
  Pk_recon0 = new double[nk * 10];

  for (int i = 0; i < 10 * nk; ++i) {
    Pk_tar[i] = 0.0;
    Pk_tar0[i] = 0.0;
    Pk_recon[i] = 0.0;
    Pk_recon0[i] = 0.0;
  }

  for (int i = myrank; i < nk; i += numprocs) {
    ki = k[i];

    Pk_tar[0 + 10 * i] = Spectra.Preg_2loop(DENS, DENS, ki);
    Pk_tar[1 + 10 * i] = Spectra.Preg_2loop(DENS, VELO, ki);
    Pk_tar[2 + 10 * i] = Spectra.Preg_2loop(VELO, VELO, ki);

    res_A = Direct_red.get_Aterm(ki);

    Pk_tar[3 + 10 * i] = res_A["A2"];
    Pk_tar[4 + 10 * i] = res_A["A4"];
    Pk_tar[5 + 10 * i] = res_A["A6"];

    res_B = Direct_red.get_Bterm(ki);

    Pk_tar[6 + 10 * i] = res_B["B2"];
    Pk_tar[7 + 10 * i] = res_B["B4"];
    Pk_tar[8 + 10 * i] = res_B["B6"];
    Pk_tar[9 + 10 * i] = res_B["B8"];

    res_Pk = Fast_spectra.get_spectra_2l(ki);

    Pk_recon[0 + 10 * i] = res_Pk["dd"];
    Pk_recon[1 + 10 * i] = res_Pk["dt"];
    Pk_recon[2 + 10 * i] = res_Pk["tt"];

    res_A = Fast_spectra.get_Aterm(ki);

    Pk_recon[3 + 10 * i] = res_A["A2"];
    Pk_recon[4 + 10 * i] = res_A["A4"];
    Pk_recon[5 + 10 * i] = res_A["A6"];

    res_B = Fast_spectra.get_Bterm(ki);

    Pk_recon[6 + 10 * i] = res_B["B2"];
    Pk_recon[7 + 10 * i] = res_B["B4"];
    Pk_recon[8 + 10 * i] = res_B["B6"];
    Pk_recon[9 + 10 * i] = res_B["B8"];
  }

  all_reduce(world, Pk_tar, nk * 10, Pk_tar0, plus<double>());
  all_reduce(world, Pk_recon, nk * 10, Pk_recon0, plus<double>());

  if (myrank == 0) {
    sprintf(fname, "%s_tarPk.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e %e %e %e\n", k[i], Cosmo.Plin(k[i]),
              Cosmo.Pno_wiggle(k[i]), Pk_tar0[0 + 10 * i], Pk_tar0[1 + 10 * i],
              Pk_tar0[2 + 10 * i]);
    }

    fclose(fp);

    sprintf(fname, "%s_tarA.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e %e\n", k[i], Pk_tar0[3 + 10 * i],
              Pk_tar0[4 + 10 * i], Pk_tar0[5 + 10 * i]);
    }

    fclose(fp);

    sprintf(fname, "%s_tarB.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e %e %e\n", k[i], Pk_tar0[6 + 10 * i],
              Pk_tar0[7 + 10 * i], Pk_tar0[8 + 10 * i], Pk_tar0[9 + 10 * i]);
    }

    fclose(fp);

    sprintf(fname, "%s_reconPk.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e %e %e %e\n", k[i], Cosmo.Plin(k[i]),
              Cosmo.Pno_wiggle(k[i]), Pk_recon0[0 + 10 * i],
              Pk_recon0[1 + 10 * i], Pk_recon0[2 + 10 * i]);
    }

    fclose(fp);

    sprintf(fname, "%s_reconA.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e %e\n", k[i], Pk_recon0[3 + 10 * i],
              Pk_recon0[4 + 10 * i], Pk_recon0[5 + 10 * i]);
    }

    fclose(fp);

    sprintf(fname, "%s_reconB.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i = 0; i < nk; ++i) {
      fprintf(fp, "%e %e %e %e %e\n", k[i], Pk_recon0[6 + 10 * i],
              Pk_recon0[7 + 10 * i], Pk_recon0[8 + 10 * i],
              Pk_recon0[9 + 10 * i]);
    }

    fclose(fp);
  }

  delete[] Pk_tar;
  delete[] Pk_tar0;
  delete[] Pk_recon;
  delete[] Pk_recon0;

  return;
}

void validate_recon_Bk(char *inifile, char *header, mpi::communicator &world) {
  int myrank, numprocs;

  myrank = world.rank();
  numprocs = world.size();

  if (myrank == 0)
    cout << "[NOTE] MPI parallel mode" << endl;
  world.barrier();
  cout << "myrank/numprocs: " << myrank << "/" << numprocs << endl;

  FILE *fp;
  char fname[256];
  double ki, Bki, k1, k2, k3;
  double *Bk_tar, *Bk_tar0, *Bk_recon, *Bk_recon0;
  double kiso1[] = {0.015, 0.035, 0.055, 0.075, 0.095};
  double kiso2[] = {0.075, 0.095, 0.115, 0.135, 0.155};

  const int mode = 0;
  const int nk = 100;
  double k[nk];
  const double kmin = 1e-3;
  const double kmax = 0.6;

  params Params(inifile);
  // Params.dparams["z"] = 0.900902;
  // Params.sparams["fast_fidmodels_config"] = "config_all.dat";
  // Params.sparams["fast_fidmodels_config"] = "config_bispec_all.dat";
  Params.sparams["fast_fidmodels_config"] = "config_bispec.dat";
  // Params.bparams["direct_SPT"] = true;
  // sprintf(fname, "recon_test/data/z1/%s", argv[2]);
  // sprintf(header, "recon_test/data/z3/%s", argv[2]);

  if (mode == 0) {
    Params.dparams["z"] = 0.0;
    // sprintf(header, "recon_test/data/z0/%s", argv[2]);
    // Params.dparams["z"] = 0.0;
    // sprintf(fname, "recon_test/data/z0/%s", argv[2]);
  } else if (mode == 1) {
    Params.dparams["z"] = 0.5;
    // sprintf(header, "recon_test/data/z05/%s", argv[2]);
    // Params.dparams["z"] = 0.520732;
    // sprintf(fname, "recon_test/data/z05/%s", argv[2]);
  } else if (mode == 2) {
    Params.dparams["z"] = 1.0;
    // sprintf(header, "recon_test/data/z1/%s", argv[2]);
    // Params.dparams["z"] = 0.900902;
    // sprintf(fname, "recon_test/data/z1/%s", argv[2]);
  } else if (mode == 3) {
    Params.dparams["z"] = 2.0;
    // sprintf(header, "recon_test/data/z2/%s", argv[2]);
    // Params.dparams["z"] = 2.11735;
    // sprintf(fname, "recon_test/data/z2/%s", argv[2]);
  } else if (mode == 4) {
    Params.dparams["z"] = 3.0;
    // sprintf(header, "recon_test/data/z3/%s", argv[2]);
    // Params.dparams["z"] = 3.1272;
    // sprintf(fname, "recon_test/data/z3/%s", argv[2]);
  }

  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);
  fast_bispectra Fast_bispectra(Params, Cosmo, Spectra);
  bispectra Bispectra(Params, Cosmo, Spectra);

  if ((fp = fopen(fname, "wb")) == NULL) {
    cerr << "file open error" << endl;
    exit(1);
  }

  for (int i = 0; i < nk; ++i) {
    k[i] = (log(kmax) - log(kmin)) / (double(nk) - 1.0) * i + log(kmin);
    k[i] = exp(k[i]);
  }

  Bk_tar = new double[nk * 14];
  Bk_tar0 = new double[nk * 14];
  Bk_recon = new double[nk * 14];
  Bk_recon0 = new double[nk * 14];

  for (int i = 0; i < nk * 14; ++i) {
    Bk_tar[i] = 0.0;
    Bk_tar0[i] = 0.0;
    Bk_recon[i] = 0.0;
    Bk_recon0[i] = 0.0;
  }

  for (int i1 = 0; i1 < nk; ++i1) {
    ki = k[i1];
    for (int i2 = 0; i2 < 14; ++i2) {
      if ((i2 + 14 * i1) % numprocs != myrank)
        continue;

      if (i2 < 4) {
        k1 = ki;
        k2 = ki;
        k3 = ki;
      } else if (4 <= i2 && i2 < 9) {
        k1 = ki;
        k2 = ki;
        k3 = kiso1[i2 - 4];
      } else {
        k1 = ki;
        k2 = kiso2[i2 - 9];
        k3 = kiso2[i2 - 9];
      }

      if (i2 == 1) {
        Bk_recon[i2 + 14 * i1] =
            Fast_bispectra.get_bispectrum(DENS, DENS, VELO, k1, k2, k3);
        Bk_tar[i2 + 14 * i1] =
            Bispectra.Bispec_1loop(DENS, DENS, VELO, k1, k2, k3);
      } else if (i2 == 2) {
        Bk_recon[i2 + 14 * i1] =
            Fast_bispectra.get_bispectrum(DENS, VELO, VELO, k1, k2, k3);
        Bk_tar[i2 + 14 * i1] =
            Bispectra.Bispec_1loop(DENS, VELO, VELO, k1, k2, k3);
      } else if (i2 == 3) {
        Bk_recon[i2 + 14 * i1] =
            Fast_bispectra.get_bispectrum(VELO, VELO, VELO, k1, k2, k3);
        Bk_tar[i2 + 14 * i1] =
            Bispectra.Bispec_1loop(VELO, VELO, VELO, k1, k2, k3);
      } else {
        Bk_recon[i2 + 14 * i1] =
            Fast_bispectra.get_bispectrum(DENS, DENS, DENS, k1, k2, k3);
        Bk_tar[i2 + 14 * i1] =
            Bispectra.Bispec_1loop(DENS, DENS, DENS, k1, k2, k3);
      }
      printf("%e %e %e\n", ki, Bk_tar[i2 + 14 * i1], Bk_recon[i2 + 14 * i1]);
    }
  }

  all_reduce(world, Bk_tar, nk * 14, Bk_tar0, plus<double>());
  all_reduce(world, Bk_recon, nk * 14, Bk_recon0, plus<double>());

  if (myrank == 0) {
    sprintf(fname, "%s_tarBkeq.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i1 = 0; i1 < nk; ++i1) {
      fprintf(fp, "%e %e %e %e %e\n", k[i1], Bk_tar0[0 + 14 * i1],
              Bk_tar0[1 + 14 * i1], Bk_tar0[2 + 14 * i1], Bk_tar0[3 + 14 * i1]);
    }

    fclose(fp);

    sprintf(fname, "%s_reconBkeq.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i1 = 0; i1 < nk; ++i1) {
      fprintf(fp, "%e %e %e %e %e\n", k[i1], Bk_recon0[0 + 14 * i1],
              Bk_recon0[1 + 14 * i1], Bk_recon0[2 + 14 * i1],
              Bk_recon0[3 + 14 * i1]);
    }

    fclose(fp);

    sprintf(fname, "%s_tarBkiso1.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i1 = 0; i1 < nk; ++i1) {
      fprintf(fp, "%e %e %e %e %e %e\n", k[i1], Bk_tar0[4 + 14 * i1],
              Bk_tar0[5 + 14 * i1], Bk_tar0[6 + 14 * i1], Bk_tar0[7 + 14 * i1],
              Bk_tar0[8 + 14 * i1]);
    }

    fclose(fp);

    sprintf(fname, "%s_reconBkiso1.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i1 = 0; i1 < nk; ++i1) {
      fprintf(fp, "%e %e %e %e %e %e\n", k[i1], Bk_recon0[4 + 14 * i1],
              Bk_recon0[5 + 14 * i1], Bk_recon0[6 + 14 * i1],
              Bk_recon0[7 + 14 * i1], Bk_recon0[8 + 14 * i1]);
    }

    fclose(fp);

    sprintf(fname, "%s_tarBkiso2.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i1 = 0; i1 < nk; ++i1) {
      fprintf(fp, "%e %e %e %e %e %e\n", k[i1], Bk_tar0[9 + 14 * i1],
              Bk_tar0[10 + 14 * i1], Bk_tar0[11 + 14 * i1],
              Bk_tar0[12 + 14 * i1], Bk_tar0[13 + 14 * i1]);
    }

    fclose(fp);

    sprintf(fname, "%s_reconBkiso2.dat", header);

    if ((fp = fopen(fname, "w")) == NULL) {
      printf("File open error!:%s\n", fname);
      exit(1);
    }

    for (int i1 = 0; i1 < nk; ++i1) {
      fprintf(fp, "%e %e %e %e %e %e\n", k[i1], Bk_recon0[9 + 14 * i1],
              Bk_recon0[10 + 14 * i1], Bk_recon0[11 + 14 * i1],
              Bk_recon0[12 + 14 * i1], Bk_recon0[13 + 14 * i1]);
    }

    fclose(fp);
  }

  delete[] Bk_tar;
  delete[] Bk_tar0;
  delete[] Bk_recon;
  delete[] Bk_recon0;

  return;
}

void validate_binnedBk(char *inifile, char *fname) {
  chrono::system_clock::time_point start, end;
  double elapsed;
  const int nk = 30;
  const double dk = 0.01;
  int nkbin;
  double k1, k2, k3;
  map<string, vector<double>> kbin;
  vector<double> k1min, k1max, k2min, k2max, k3min, k3max, res;
  FILE *fp;

  params Params(inifile);
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);

  for (int i1 = 0; i1 < nk; ++i1) {
    for (int i2 = i1; i2 < nk; ++i2) {
      for (int i3 = i2; i3 < nk; ++i3) {
        k1min.push_back(dk * i1);
        k1max.push_back(dk * (i1 + 1));
        k2min.push_back(dk * i2);
        k2max.push_back(dk * (i2 + 1));
        k3min.push_back(dk * i3);
        k3max.push_back(dk * (i3 + 1));
      }
    }
  }

  kbin["k1min"] = k1min;
  kbin["k1max"] = k1max;
  kbin["k2min"] = k2min;
  kbin["k2max"] = k2max;
  kbin["k3min"] = k3min;
  kbin["k3max"] = k3max;

  nkbin = k1min.size();
  printf("kbin set:%d\n", nkbin);

  start = chrono::system_clock::now();
  fast_bispectra Fast_bispectra(Params, Cosmo, Spectra);
  end = chrono::system_clock::now();
  elapsed =
      chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "Fast bispectra:" << elapsed << "[ms]" << endl;

  start = chrono::system_clock::now();
  res = Fast_bispectra.get_binned_bispectrum(DENS, DENS, DENS, kbin);
  end = chrono::system_clock::now();
  elapsed =
      chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "binned bispectra:" << elapsed << "[ms]" << endl;

  if ((fp = fopen(fname, "w")) == NULL) {
    cerr << "file open error" << endl;
    exit(1);
  }

  for (int i = 0; i < nkbin; ++i) {
    k1 = 0.5 * (k1min[i] + k1max[i]);
    k2 = 0.5 * (k2min[i] + k2max[i]);
    k3 = 0.5 * (k3min[i] + k3max[i]);

    fprintf(fp, "%e %e %e %e\n", k1, k2, k3, res[i]);
  }

  fclose(fp);

  return;
}

void validate_Bk_halofit(char *inifile, char *fname) {
  double Bki, ki, k1, k2, k3;
  double kiso1[] = {0.015, 0.035, 0.055, 0.075, 0.095};
  double kiso2[] = {0.075, 0.095, 0.115, 0.135, 0.155};
  FILE *fp;

  const double kmin = 0.01;
  const double kmax = 0.3;
  const int nk = 30;

  params Params(inifile);
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);
  nonlinear Nonlinear(Params, Cosmo);

  if ((fp = fopen(fname, "w")) == NULL) {
    cerr << "file open error" << endl;
    exit(1);
  }

  for (int i = 0; i < nk; ++i) {
    ki = (kmax - kmin) / (nk - 1.0) * (i + 0.5);
    fprintf(fp, "%e %e", ki, Nonlinear.Pk_halofit(ki));
    for (int j = 0; j < 11; ++j) {
      if (j == 0) {
        k1 = ki;
        k2 = ki;
        k3 = ki;
      } else if (1 <= j && j < 6) {
        k1 = ki;
        k2 = ki;
        k3 = kiso1[j - 1];
      } else {
        k1 = ki;
        k2 = kiso2[j - 6];
        k3 = kiso2[j - 6];
      }
      Bki = Nonlinear.Bk_halofit(k1, k2, k3);
      fprintf(fp, " %e", Bki);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);

  return;
}
