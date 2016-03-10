#include "io.h"

#include <iostream>

namespace deeplearning {
namespace embedding {
namespace util {

template <typename BasicType>
void WriteBasicItem(std::ostream* out, BasicType item) {
  out->write((const char*)&item, sizeof(BasicType));
}

template <typename BasicType>
void ReadBasicItem(std::istream* in, BasicType* item) {
  in->read((char*)item, sizeof(BasicType));
}

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning
