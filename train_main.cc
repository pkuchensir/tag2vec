#include <eigen3/Eigen/Core>
#include <fstream>
#include <omp.h>
#include <string>

#include "document/standard_document_iterator.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {
namespace {

void PrintScoreItemVec(const std::vector<ScoreItem>& score_item_vec) {
  for (const ScoreItem& score_item : score_item_vec) {
    LOG(INFO) << score_item.ToString();
  }
}

void Train(const std::string& input_path, const std::string& model_path) {
  Tag2Vec tag2vec;
  std::ifstream fin(input_path);
  StandardDocumentIterator iterator(&fin);
  tag2vec.Train(&iterator, 10);

  std::string tag = "music";
  LOG(INFO) << tag << ":";
  std::vector<ScoreItem> ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 20);
  PrintScoreItemVec(ans);

  tag = "food";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 20);
  PrintScoreItemVec(ans);

  /*
  std::ofstream fout(model_path);
  tag2vec.Write(&fout);
  */
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  Eigen::initParallel();
  omp_set_num_threads(4);
  deeplearning::embedding::Train("/home/dhuang/twitter.big.train.txt",
                                 "/home/dhuang/tag2vec.model");
  return 0;
}
