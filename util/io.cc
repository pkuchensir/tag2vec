#include "io.h"

#include <iostream>
#include <string>

void WriteString(std::ostream* out, const std::string& str) {
  out->write(str.c_str(), str.size() + 1);
}

void ReadString(std::istream* in, std::string* str) {
  std::getline(*in, *str, '\0');
}
