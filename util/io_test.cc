#include "util/io.h"

#include <iostream>
#include <sstream>

using namespace std;


int main(int argc, char** argv) {
  // int i = 100;
  // string i = "abcd";
  double i = 1;
  ostringstream oss;
  deeplearning::embedding::util::WriteBasicItem(&oss, i);
  /*
  int j = 10;
  istringstream iss(oss.str());
  deeplearning::embedding::util::ReadBasicItem(&iss, &j);
  cout << j << endl;
  */


  return 0;
}
