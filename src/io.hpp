#ifndef IO_HEADER_INCLUDED
#define IO_HEADER_INCLUDED

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <vector>


using namespace std;

enum TYPE{
  INT,
  DOUBLE,
  STRING,
  BOOL,
};

class params{
private:
  vector<string> name_list;
  map<string, TYPE> type_list;
  void initialize(void);
  void set_default_parameter(void);
  void load_parameter(const char *ini_fname);
  void store_parameter(string pname, string p);
  void check_conflict(void);
public:
  params(void);
  params(char *params_fname);
  map<string, int> iparams; // (int) parameters related with integration
  map<string, double> dparams; // (double) parameters related with integration
  map<string, string> sparams;
  map<string, bool> bparams;
  bool flag_transfer_from_file, flag_transfer_EH;
  void show_parameter(void);
};
#endif
