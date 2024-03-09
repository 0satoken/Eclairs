#ifndef PARAMS_HEADER_INCLUDED
#define PARAMS_HEADER_INCLUDED

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

enum TYPE {
  INT,
  DOUBLE,
  STRING,
  BOOL,
};

class params {
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
  map<string, int> iparams;
  map<string, double> dparams;
  map<string, string> sparams;
  map<string, bool> bparams;
  void show_parameter(void);
  vector<string> get_name_list(void);
  map<string, TYPE> get_type_list(void);
};
#endif
