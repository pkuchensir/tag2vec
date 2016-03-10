#ifndef UTIL_IO_H_
#define UTIL_IO_H_

#include <iostream>
#include <string>
#include <vector>

namespace deeplearning {
namespace embedding {
namespace util {

template <typename BasicType>
void WriteBasicItem(std::ostream* out, BasicType item);
template <typename BasicType>
void ReadBasicItem(std::istream* in, BasicType* item);

void WriteString(std::ostream* out, const std::string& str);
void ReadString(std::istream* in, std::string* str);

template <typename BasicType>
void WriteBasicItemVector(std::ostream* out, const std::vector<BasicType>& v);
template <typename BasicType>
void ReadBasicItemVector(std::istream* in, std::vector<BasicType>* v);

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

#include "util/io.incl.h"

#endif  // UTIL_IO_H_
