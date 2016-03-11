#include "model/tag2vec.h"

#include <eigen3/Eigen/Core>
#include <iostream>
#include <omp.h>
#include <string>

#include "document/document.h"
#include "util/io.h"

namespace deeplearning {
namespace embedding {

void Tag2Vec::Train(DocumentIterator* iterator, size_t iter) {}

Eigen::RowVectorXf Tag2Vec::Infer(const std::vector<std::string>& words,
                                  size_t iter) const {
  Eigen::RowVectorXf ans;
  return ans;
}

void Tag2Vec::Write(std::ostream* out) {
  util::WriteBasicItem(out, layer_size_);
  util::WriteBasicItem(out, min_count_);
  util::WriteBasicItem(out, sample_);
  util::WriteBasicItem(out, init_alpha_);
  util::WriteBasicItem(out, min_alpha_);
}

void Tag2Vec::Read(std::istream* in, Tag2Vec* tag2vec) {
  util::ReadBasicItem(in, &tag2vec->layer_size_);
  util::ReadBasicItem(in, &tag2vec->min_count_);
  util::ReadBasicItem(in, &tag2vec->sample_);
  util::ReadBasicItem(in, &tag2vec->init_alpha_);
  util::ReadBasicItem(in, &tag2vec->min_alpha_);
}

std::string Tag2Vec::ConfigString() const {
  std::string ans;
  ans += "layer_size=" + std::to_string(layer_size_) + ";";
  ans += "min_count=" + std::to_string(min_count_) + ";";
  ans += "sample=" + std::to_string(sample_) + ";";
  ans += "init_alpha=" + std::to_string(init_alpha_) + ";";
  ans += "min_alpha=" + std::to_string(min_alpha_) + ";";
  return ans;
}

}  // namespace embedding
}  // namespace deeplearning
