#ifndef UTIL_IO_H_
#define UTIL_IO_H_

#include <iostream>
#include <string>

namespace deeplearning {
namespace embedding {
namespace util {

template <typename BasicType>
void WriteBasicItem(std::ostream* out, BasicType item);
template <typename BasicType>
void ReadBasicItem(std::istream* in, BasicType* item);

void WriteString(std::ostream* out, const std::string& str);
void ReadString(std::istream* in, std::string* str);

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

#include "util/io.incl.h"

#endif  // UTIL_IO_H_