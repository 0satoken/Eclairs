#include <iostream>
#include <string>
#include <fstream>
#include "kernel.hpp"
#include "vector.hpp"
#include "io.hpp"
#include "cosmology.hpp"
#include "spectra.hpp"


using namespace std;

void run(void);
void run(char *params_fname);

int main(int argc, char *argv[]){
  if(argc != 1 && argc != 2){
    cerr << "Usage:" << argv[0] << " [parameter file]" << endl;
    exit(1);
  }


  cout << "\"Eclairs\" starts calculation." << endl;

  if(argc == 1){
    run();
  }
  else{
    run(argv[1]);
  }

  return 0;
}

void run(char *params_fname){
  params Params(params_fname);
  cosmology Cosmo(Params);
  spectra Spectra(Cosmo);

  return;
}

void run(void){
  params Params;
  cosmology Cosmo(Params);
  spectra Spectra(Cosmo);

  return;
}
