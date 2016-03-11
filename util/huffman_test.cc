#include "util/huffman.h"

#include <string>
#include <vector>

#include "document/vocabulary.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {
namespace {

void TestHuffman() {
  std::vector<Vocabulary::Item*> items = {
      new Vocabulary::Item(0, 7, "a"), new Vocabulary::Item(1, 5, "b"),
      new Vocabulary::Item(2, 6, "c"), new Vocabulary::Item(3, 5, "d"),
      new Vocabulary::Item(4, 10, "e")};
  Huffman huffman;
  huffman.Build(items);

  std::string expect =
      "10: 3,1,\n"
      "010: 3,2,0,\n"
      "11: 3,1,\n"
      "011: 3,2,0,\n"
      "00: 3,2,\n";
  CHECK_EQ(expect, huffman.ToString());
}

}  // namespace
}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::util::TestHuffman();
  LOG(INFO) << "PASS";
  return 0;
}
