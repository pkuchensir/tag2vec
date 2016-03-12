#include "model/tag2vec.h"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "model/tag2vec_helper.h"
#include "util/io.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {

Tag2Vec::Tag2Vec() {
  Initialize();
}

Tag2Vec::Tag2Vec(size_t layer_size, size_t min_count, float sample,
                 float init_alpha, float min_alpha)
    : layer_size_(layer_size),
      min_count_(min_count),
      sample_(sample),
      init_alpha_(init_alpha),
      min_alpha_(min_alpha) {
  Initialize();
}

Tag2Vec::~Tag2Vec() {
  delete random_;
}

void Tag2Vec::Train(std::vector<const Document*>* documents, size_t iter) {
  CHECK(!has_trained_) << "Tag2Vec has already been trained.";
  for (size_t t = 0; t < iter; ++t) {
    std::shuffle(documents->begin(), documents->end(), random_->engine());
  }
  has_trained_ = true;
}

void Tag2Vec::Train(DocumentIterator* iterator, size_t iter) {
  CHECK(!has_trained_) << "Tag2Vec has already been trained.";
  std::vector<const Document*> documents;
  Document* document;
  while ((document = iterator->NextDocument())) {
    documents.push_back(document);
  }
  Train(&documents, iter);
  for (const Document* document : documents) {
    delete document;
  }
  has_trained_ = true;
}

Eigen::RowVectorXf Tag2Vec::Infer(const std::vector<std::string>& words,
                                  size_t iter) const {
  Eigen::RowVectorXf ans;
  return ans;
}

void Tag2Vec::Write(std::ostream* out) {
  CHECK(has_trained_) << "Tag2Vec has not been trained.";
  util::WriteBasicItem(out, layer_size_);
  util::WriteBasicItem(out, min_count_);
  util::WriteBasicItem(out, sample_);
  util::WriteBasicItem(out, init_alpha_);
  util::WriteBasicItem(out, min_alpha_);
}

void Tag2Vec::Read(std::istream* in, Tag2Vec* tag2vec) {
  CHECK(!tag2vec->has_trained_) << "Tag2Vec has been trained.";
  util::ReadBasicItem(in, &tag2vec->layer_size_);
  util::ReadBasicItem(in, &tag2vec->min_count_);
  util::ReadBasicItem(in, &tag2vec->sample_);
  util::ReadBasicItem(in, &tag2vec->init_alpha_);
  util::ReadBasicItem(in, &tag2vec->min_alpha_);
  tag2vec->Initialize();
  tag2vec->has_trained_ = true;
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

void Tag2Vec::Initialize() {
  random_ = new Random;
}

}  // namespace embedding
}  // namespace deeplearning
