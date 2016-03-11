#include "document/word.h"

#include <sstream>
#include <string>
#include <vector>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace {

void CheckEqWord(const Word& expect, const Word& actual) {
  CHECK_EQ(expect.index(), actual.index()) << "index mismatched.";
  CHECK_EQ(expect.count(), actual.count()) << "count mismatched.";
  CHECK_EQ(expect.text(), actual.text()) << "text mismatched.";
  CHECK_EQ(expect.probability(), actual.probability())
      << "probability mismatched.";

  if (expect.codes() && actual.codes()) {
    CHECK_EQ(expect.codes()->size(), actual.codes()->size())
        << "Size of codes mismatched.";
    for (size_t i = 0; i < expect.codes()->size(); ++i) {
      CHECK_EQ((*expect.codes())[i], (*actual.codes())[i]) << "codes[" << i
                                                           << "] mismatched.";
    }
  } else {
    CHECK(expect.codes() == nullptr && actual.codes() == nullptr)
        << "Existence of cores mismatched.";
  }

  if (expect.points() && actual.points()) {
    CHECK_EQ(expect.points()->size(), actual.points()->size())
        << "Size of points mismatched.";
    for (size_t i = 0; i < expect.points()->size(); ++i) {
      CHECK_EQ((*expect.points())[i], (*actual.points())[i]) << "points[" << i
                                                             << "] mismatched.";
    }
  } else {
    CHECK(expect.points() == nullptr && actual.points() == nullptr)
        << "Existence of cores mismatched.";
  }
}

void TestWordIO() {
  std::ostringstream oss;
  Word w1(123, 456, "abc", 9.4);
  w1.Write(&oss);
  Word w2(456, 123, "zyx", 10.3);
  w2.Write(&oss);

  std::istringstream iss(oss.str());
  Word w1x;
  Word::Read(&iss, &w1x);
  CheckEqWord(w1, w1x);
  Word w2x;
  Word::Read(&iss, &w2x);
  CheckEqWord(w2, w2x);
}

void TestWord() {
  TestWordIO();
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::TestWord();
  LOG(INFO) << "PASS";
  return 0;
}
