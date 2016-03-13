#include <eigen3/Eigen/Core>
#include <fstream>
#include <string>
#include <vector>

#include "model/tag2vec.h"
#include "document/score_item.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace {

void PrintScoreItemVec(const std::vector<ScoreItem>& score_item_vec) {
  for (const ScoreItem& score_item : score_item_vec) {
    LOG(INFO) << score_item.ToString();
  }
}

void Test(const std::string& model_path) {
  Tag2Vec tag2vec;
  std::ifstream fin(model_path);
  Tag2Vec::Read(&fin, &tag2vec);

  std::string tag = "music";
  LOG(INFO) << tag << ":";
  std::vector<ScoreItem> ans = tag2vec.MostSimilar(tag2vec.TagVec("music"), 20);
  PrintScoreItemVec(ans);

  tag = "food";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec("music"), 20);
  PrintScoreItemVec(ans);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::Test("/home/dhuang/tag2vec.model");
  return 0;
}
