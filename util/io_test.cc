#include "util/io.h"

#include <sstream>
#include <string>
#include <vector>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {
namespace {

void TestIO() {
  std::ostringstream oss;
  float a = 123.45;
  WriteBasicItem(&oss, a);
  size_t b = 789;
  WriteBasicItem(&oss, b);
  std::string c = "abcd\nefg";
  WriteString(&oss, c);
  std::vector<int> d = {7, 6, 5, 4};
  WriteBasicItemVector(&oss, d);

  std::istringstream iss(oss.str());
  float ax;
  ReadBasicItem(&iss, &ax);
  CHECK_EQ(a, ax) << "ax mismatched.";

  size_t bx;
  ReadBasicItem(&iss, &bx);
  CHECK_EQ(b, bx) << "bx mismatched.";

  std::string cx;
  ReadString(&iss, &cx);
  CHECK_EQ(c, cx) << "cx mismatched.";

  std::vector<int> dx;
  ReadBasicItemVector(&iss, &dx);
  CHECK_EQ(d.size(), dx.size()) << "Size of dx mismatched.";
  for (size_t i = 0; i < d.size(); ++i) {
    CHECK_EQ(d[i], dx[i]) << "dx[" << i << "] mismatched.";
  }
}

}  // namespace
}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::util::TestIO();
  LOG(INFO) << "PASS";
  return 0;
}
