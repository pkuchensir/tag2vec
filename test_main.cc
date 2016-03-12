#include <eigen3/Eigen/Core>
#include <fstream>
#include <string>

// #include "document/standard_document_iterator.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {
namespace {

void Test(const std::string& model_path) {
  Tag2Vec tag2vec;
  std::ifstream fin(model_path);
  Tag2Vec::Read(&fin, &tag2vec);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  // Eigen::initParallel();
  deeplearning::embedding::Test("/home/dhuang/tag2vec.model");
  return 0;
}
