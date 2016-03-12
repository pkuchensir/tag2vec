#include "util/io.h"

#include <eigen3/Eigen/Core>
#include <sstream>
#include <string>
#include <vector>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {
namespace {

void TestNormalIO() {
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

void TestMatrixIO() {
  std::ostringstream oss;
  Eigen::MatrixXf expect(4, 3);
  expect << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  WriteMatrix(&oss, expect);

  std::istringstream iss(oss.str());
  Eigen::MatrixXf actual;
  ReadMatrix(&iss, &actual);
  CHECK_EQ(expect, actual) << "Matrix mismatched.";
}

void TestIO() {
  TestNormalIO();
  TestMatrixIO();
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
