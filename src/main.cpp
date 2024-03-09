#include "cosmology.hpp"
#include "direct_red.hpp"
#include "params.hpp"
#include "spectra.hpp"
#include <fstream>
#include <iostream>
#include <string>

#ifdef MPI_PARALLEL
#include "validation.hpp"
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

using namespace std;

void run(void);
void run(char *params_fname);

int main(int argc, char *argv[]) {
  /*
  #ifdef MPI_PARALLEL
    mpi::environment env(argc, argv);
    mpi::communicator world;

    if (world.rank() == 0)
      cout << "[NOTE] MPI parallel mode" << endl;
    world.barrier();
    cout << "myrank/numprocs: " << world.rank() << "/" << world.size() << endl;

    // validate_Pk(argv[1], argv[2], world);
    // validate_Bk(argv[1], argv[2], world);
    // validate_recon_Pk(argv[1], argv[2], world);
    // validate_Pk_red(argv[1], argv[2], world);

    return 0;
  #endif
  */

  if (argc != 1 && argc != 2) {
    cerr << "Usage:" << argv[0] << " [parameter file]" << endl;
    exit(1);
  }

  cout << "\"Eclairs\" starts calculation." << endl;

  if (argc == 1) {
    run();
  } else {
    run(argv[1]);
  }

  return 0;
}

void run(char *params_fname) {
  params Params(params_fname);
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);

#ifdef MPI_PARALLEL
  mpi::environment env;
  mpi::communicator world;
  int myrank, numprocs;

  myrank = world.rank();
  numprocs = world.size();

  if (myrank == 0)
    cout << "[NOTE] MPI parallel mode" << endl;
  world.barrier();
  cout << "myrank/numprocs: " << world.rank() << "/" << world.size() << endl;

  const int nk = 500;
  const double kmin = 1e-3;
  const double kmax = 5.0;
  double ki, buf;
  map<string, double> res_A, res_B;
  double *k, *Pk, *Pk0;
  FILE *fp;

  k = new double[nk];
  Pk = new double[7 * nk];
  Pk0 = new double[7 * nk];

  for (int i = 0; i < nk; ++i) {
    k[i] = (log(kmax) - log(kmin)) / (nk - 1.0) * i + log(kmin);
    k[i] = exp(k[i]);
    for (int j = 0; j < 7; ++j) {
      Pk[j + 7 * i] = 0.0;
      Pk0[j + 7 * i] = 0.0;
    }
  }

  for (int i = myrank; i < nk; i += numprocs) {
    ki = k[i];

    res_A = Direct_red.get_Aterm(ki);

    Pk[0 + 7 * i] = res_A["A2"];
    Pk[1 + 7 * i] = res_A["A4"];
    Pk[2 + 7 * i] = res_A["A6"];

    res_B = Direct_red.get_Bterm(ki);

    Pk[3 + 7 * i] = res_B["B2"];
    Pk[4 + 7 * i] = res_B["B4"];
    Pk[5 + 7 * i] = res_B["B6"];
    Pk[6 + 7 * i] = res_B["B8"];
  }

  all_reduce(world, Pk, 7 * nk, Pk0, plus<double>());

  if (myrank == 0) {
    if ((fp = fopen("TNS.dat", "wb")) == NULL) {
      cerr << "file open error" << endl;
      exit(1);
    }

    fwrite(&nk, 1, sizeof(int), fp);

    for (int i = 0; i < nk; ++i) {
      buf = k[i];
      fwrite(&buf, 1, sizeof(double), fp);
      for (int j = 0; j < 7; ++j){
        buf = Pk0[j + 7 * i];
        fwrite(&buf, 1, sizeof(double), fp);
      }
    }

    fwrite(&nk, 1, sizeof(int), fp);

    fclose(fp);


    printf("#k A2 A4 A6 B2 B4 B6 B8\n");

    for (int i = 0; i < nk; ++i) {
      printf("%e", k[i]);
      for (int j = 0; j < 7; ++j)
        printf(" %e", Pk0[j + 7 * i]);
      printf("\n");
    }

  }

  delete[] k;
  delete[] Pk;
  delete[] Pk0;
#endif

  return;
}

void run(void) {
  params Params;
  cosmology Cosmo(Params);
  spectra Spectra(Params, Cosmo);

  return;
}
