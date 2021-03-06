#include "document/vocabulary.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace {

void BuildVocabulary(const std::vector<std::string>& items, size_t min_count,
                     Vocabulary* vocabulary) {
  for (const std::string& item : items) {
    vocabulary->AddItem<Vocabulary::Item>(item);
  }
  vocabulary->Build(min_count);
}

void CheckVocabulary(
    const Vocabulary& vocabulary,
    const std::vector<std::pair<std::string, size_t>>& item_count_vec) {
  size_t total_count = 0;
  for (const auto& item_count : item_count_vec) {
    total_count += item_count.second;
  }
  CHECK_EQ(total_count, vocabulary.num_retained()) << "num_retained mismatched.";
  CHECK_EQ(item_count_vec.size(), vocabulary.items_size())
      << "items_size mismatched.";
  for (size_t i = 0; i < item_count_vec.size(); ++i) {
    const Vocabulary::Item* item = vocabulary.item(i);
    CHECK_EQ(i, item->index()) << "index mismatched.";
    CHECK_EQ(item_count_vec[i].first, item->text()) << "text mismatched.";
    CHECK_EQ(item_count_vec[i].second, item->count()) << "count mismatched.";
    CHECK_EQ(item, vocabulary.item(item->text())) << "item mismatched.";
  }
}

void TestVocabularyBuild() {
  Vocabulary vocabulary;
  BuildVocabulary({"b", "c", "b", "a", "c", "c"}, 0, &vocabulary);
  std::vector<std::pair<std::string, size_t>> item_count_vec = {
      std::make_pair("c", 3), std::make_pair("b", 2), std::make_pair("a", 1)};
  CHECK_EQ(6, vocabulary.num_original()) << "num_original mismatched.";
  CheckVocabulary(vocabulary, item_count_vec);

  vocabulary.Clear();
  BuildVocabulary({"b", "c", "b", "a", "c", "c"}, 2, &vocabulary);
  item_count_vec = {std::make_pair("c", 3), std::make_pair("b", 2)};
  CHECK_EQ(6, vocabulary.num_original()) << "num_original mismatched.";
  CheckVocabulary(vocabulary, item_count_vec);
}

void TestVocabularyIO() {
  std::ostringstream oss;
  Vocabulary vocabulary;
  BuildVocabulary({"b", "c", "b", "a", "c", "c"}, 0, &vocabulary);
  vocabulary.Write(&oss);

  std::istringstream iss(oss.str());
  Vocabulary read_vocabulary;
  Vocabulary::Read<Vocabulary::Item>(&iss, &read_vocabulary);
  std::vector<std::pair<std::string, size_t>> item_count_vec = {
      std::make_pair("c", 3), std::make_pair("b", 2), std::make_pair("a", 1)};
  CHECK_EQ(6, vocabulary.num_original()) << "num_original mismatched.";
  CheckVocabulary(read_vocabulary, item_count_vec);
}

void TestVocabulary() {
  TestVocabularyBuild();
  TestVocabularyIO();
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::TestVocabulary();
  LOG(INFO) << "PASS";
  return 0;
}
