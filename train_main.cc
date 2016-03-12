#include <eigen3/Eigen/Core>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>

#include "document/standard_document_iterator.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {
namespace {

void Train(const std::string& input_path, const std::string& model_path) {
  Tag2Vec tag2vec;
  std::ifstream fin(input_path);
  StandardDocumentIterator iterator(&fin);
  tag2vec.Train(&iterator, 2);

  std::ofstream fout(model_path);
  tag2vec.Write(&fout);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  Eigen::initParallel();
  deeplearning::embedding::Train("/home/dhuang/twitter.big.train.txt",
                                 "/home/dhuang/tag2vec.model");
  return 0;
}
