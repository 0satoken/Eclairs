#ifndef VALIDATION_HEADER_INCLUDED
#define VALIDATION_HEADER_INCLUDED

#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
namespace mpi = boost::mpi;

using namespace std;

void validate_Pk(char *inifile, char *fname, mpi::communicator &world);
void validate_Pk_TNS(char *inifile, char *fname, mpi::communicator &world);
void validate_Pk_red(char *inifile, char *fname, mpi::communicator &world);
void validate_Bk(char *inifile, char *fname, mpi::communicator &world);
void validate_recon_Pk(char *inifile, char *header, mpi::communicator &world);
void validate_recon_Bk(char *inifile, char *header, mpi::communicator &world);
void validate_binnedBk(char *inifile, char *fname);

#endif
