#include "util/evaluate.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "document/score_item.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {

void Evaluate(Model* model, DocumentIterator* test_data) {
  const Document* document;
  size_t num_correct = 0, num_total = 0;
  size_t count = 0;
  while ((document = test_data->NextDocument())) {
    std::unordered_set<std::string> tag_set(document->tags().begin(),
                                            document->tags().end());
    std::vector<ScoreItem> suggested_tags = model->Suggest(document->words());
    CHECK_EQ(1, suggested_tags.size()) << "Should only suggest 1 item.";
    if (tag_set.find(suggested_tags[0].item()) != tag_set.end()) {
      ++num_correct;
    }
    num_total += document->tags().size();
    delete document;
    ++count;
    if (count % 1000 == 0) {
      LOG(INFO) << "Tested " << count << ". p@1=" << (float)num_correct / num_total;
    }
  }
  LOG(INFO) << "Tested " << count << ". p@1=" << (float)num_correct / num_total;
}

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning
