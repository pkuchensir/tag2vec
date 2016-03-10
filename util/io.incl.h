#include "io.h"

#include <iostream>
#include <vector>

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

template <typename BasicType>
void WriteBasicItemVector(std::ostream* out, const std::vector<BasicType>& v) {
  size_t v_size = v.size();
  WriteBasicItem(out, v_size);
  for (BasicType item : v) {
    WriteBasicItem(out, item);
  }
}

template <typename BasicType>
void ReadBasicItemVector(std::istream* in, std::vector<BasicType>* v) {
  size_t v_size;
  ReadBasicItem(in, &v_size);
  v->resize(v_size);
  for (size_t i = 0; i < v_size; ++i) {
    ReadBasicItem(in, &(*v)[i]);
  }
}

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning
