#include "util/util.h"

#include <string>
#include <vector>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {
namespace {

void TestBasePath() {
  CHECK_EQ("/", BasePath("/abc"));
  CHECK_EQ("root/", BasePath("root/abc"));
  CHECK_EQ("/root/", BasePath("/root/abc"));
  CHECK_EQ("", BasePath("abc"));
}

void TestFilename() {
  CHECK_EQ("abc", Filename("/abc"));
  CHECK_EQ("abc", Filename("root/abc"));
  CHECK_EQ("abc", Filename("/root/abc"));
  CHECK_EQ("abc", Filename("abc"));
}

void TestJoinPath() {
  CHECK_EQ("abc", JoinPath("", "abc"));
  CHECK_EQ("/abc", JoinPath("/", "abc"));
  CHECK_EQ("root/abc", JoinPath("root", "abc"));
  CHECK_EQ("root/abc", JoinPath("root/", "abc"));
  CHECK_EQ("/root/abc", JoinPath("/root", "abc"));
  CHECK_EQ("/root/abc", JoinPath("/root/", "abc"));
}

void TestReadContent(const std::string& base_path) {
  std::string content = ReadContent(base_path + "util_test.cc");
  CHECK(content.find("TestReadContent") != std::string::npos)
      << "Failed to find TestReadContent.";

  std::string prefix = "#include \"util/util.h\"\n\n#include";
  CHECK(content.size() >= prefix.size()) << "Content is too short.";
  CHECK_EQ(prefix, content.substr(0, prefix.size()));

  std::string postfix = "return 0;\n}\n";
  CHECK(content.size() >= postfix.size()) << "Content is too short.";
  CHECK_EQ(postfix, content.substr(content.size() - postfix.size()));
}

void TestSplitCase(const std::string& str, const char* delim,
                   const std::vector<std::string>& expect) {
  const std::vector<std::string> actual = Split(str, delim);
  CHECK_EQ(expect.size(), actual.size())
      << "Number of split items does not match.";
  for (size_t i = 0; i < expect.size(); ++i) {
    CHECK_EQ(expect[i], actual[i]) << "Split item does not match.";
  }
}

void TestSplit() {
  TestSplitCase("ab", ",;", {"ab"});
  TestSplitCase("a,b,c", ",;", {"a", "b", "c"});
  TestSplitCase("a,b,c,", ",;", {"a", "b", "c"});
  TestSplitCase("a,b,c,,", ",;", {"a", "b", "c"});
  TestSplitCase(",,a,b,c,,", ",;", {"a", "b", "c"});
  TestSplitCase(",,a,;b;c,;,", ",;", {"a", "b", "c"});
  TestSplitCase("aaa,b;cc;,", ",;", {"aaa", "b", "cc"});
}

void TestTraversePath(const std::string& base_path) {
  const std::string file_to_find = "util_test.cc";
  bool found = false;
  auto check_file = [&file_to_find, &found](const std::string& full_path) {
    if (Filename(full_path) == file_to_find) {
      found = true;
    }
  };
  TraversePath(base_path + "..", check_file);
  CHECK(found) << "Failed to find " << file_to_find << ".";
}

void TestUtil(const std::string& base_path) {
  TestBasePath();
  TestFilename();
  TestJoinPath();
  TestReadContent(base_path);
  TestSplit();
  TestTraversePath(base_path);
}

}  // namespace
}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  std::string base_path = deeplearning::embedding::util::BasePath(argv[0]);
  deeplearning::embedding::util::TestUtil(base_path);
  LOG(INFO) << "PASS";
  return 0;
}
