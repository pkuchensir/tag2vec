#include "model/tag2vec.h"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/memory_document_iterator.h"
#include "document/tag.h"
#include "document/vocabulary.h"
#include "document/word.h"
#include "model/tag2vec_helper.h"
#include "util/huffman.h"
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

void Tag2Vec::Train(const std::vector<Document>& documents, size_t iter) {
  CHECK(!has_trained_) << "Tag2Vec has already been trained.";
  MemoryDocumentIterator iterator(documents);
  BuildTagVocabulary(&iterator, &tag_vocab_);
  iterator.Reset();
  BuildWordVocabulary(&iterator, min_count_, sample_, &word_vocab_);
  word_huffman_.Build(word_vocab_.items());

  CHECK(tag_vocab_.items_size() >= 1) << "Size of tag_vocab should be at least 1.";
  CHECK(word_vocab_.items_size() >= 1) << "Size of word_vocab should be at least 1.";

  LOG(INFO) << "Vocabularies for words and tags have been built.";

  // Initializes weights.
  std::uniform_real_distribution<float> dist(-0.5, 0.5);
  auto uniform = [&dist, this](size_t) { return dist(this->random_->engine()); };
  size_t tag_dim = tag_vocab_.items_size();
  tagi_ = RMatrixXf::NullaryExpr(tag_dim, layer_size_, uniform) / (float)tag_dim;
  size_t word_dim = word_vocab_.items_size() - 1;
  wordo_ = RMatrixXf::NullaryExpr(word_dim, layer_size_, uniform) / (float)word_dim;

  LOG(INFO) << "Weights have been initialized.";

  std::vector<size_t> doc_index_vec(documents.size());
  std::iota(doc_index_vec.begin(), doc_index_vec.end(), 0);
  float alpha = init_alpha_;
  size_t num_words = 0;

  for (size_t t = 0; t < iter; ++t) {
    std::shuffle(doc_index_vec.begin(), doc_index_vec.end(), random_->engine());

    #pragma omp parallel for
    for (size_t i = 0; i < doc_index_vec.size(); ++i) {
      int doc_index = doc_index_vec[i];
      const Document& document = documents[doc_index];

      // Gets words and tags.
      std::vector<const Vocabulary::Item*> word_vec, tag_vec;
      GetVocabularyItemVec(word_vocab_, document.words(), &word_vec);
      GetVocabularyItemVec(tag_vocab_, document.tags(), &tag_vec);
      DownSample(&word_vec);

      // Trains document.
      for (const Vocabulary::Item* tag : tag_vec) {
        for (const Vocabulary::Item* word : word_vec) {
          TrainSgPair(tagi_.row(tag->index()), wordo_,
                      word_huffman_.codes(word->index()),
                      word_huffman_.points(word->index()), alpha, true);
        }
      }

      #pragma omp atomic update
      num_words += document.words().size();
      const float next_alpha =
          init_alpha_ -
          (init_alpha_ - min_alpha_) * num_words / word_vocab_.num_original() / iter;
      alpha = std::max(alpha, next_alpha);

      static const size_t DISPLAY_NUM = 100000;
      if (num_words / DISPLAY_NUM >
          (num_words - document.words().size()) / DISPLAY_NUM) {
        LOG(INFO) << "Iter" << t << ": Processed " << num_words << "/"
                  << word_vocab_.num_original() * iter << " words.";
      }
    }
    LOG(INFO) << "Iteration " << t << " has finished.";
  }
  has_trained_ = true;
}

void Tag2Vec::Train(DocumentIterator* iterator, size_t iter) {
  CHECK(!has_trained_) << "Tag2Vec has already been trained.";
  std::vector<Document> documents;
  const Document* document;
  while ((document = iterator->NextDocument())) {
    documents.push_back(*document);
    delete document;
  }
  Train(documents, iter);
  has_trained_ = true;
}

Eigen::RowVectorXf Tag2Vec::Infer(const std::vector<std::string>& words,
                                  size_t iter) const {
  Eigen::RowVectorXf ans;
  return ans;
}

void Tag2Vec::Write(std::ostream* out) const {
  CHECK(has_trained_) << "Tag2Vec has not been trained.";
  util::WriteBasicItem(out, layer_size_);
  util::WriteBasicItem(out, min_count_);
  util::WriteBasicItem(out, sample_);
  util::WriteBasicItem(out, init_alpha_);
  util::WriteBasicItem(out, min_alpha_);

  word_vocab_.Write(out);
  tag_vocab_.Write(out);
  word_huffman_.Write(out);

  util::WriteMatrix(out, tagi_);
  util::WriteMatrix(out, wordo_);
}

void Tag2Vec::Read(std::istream* in, Tag2Vec* tag2vec) {
  CHECK(!tag2vec->has_trained_) << "Tag2Vec has been trained.";
  util::ReadBasicItem(in, &tag2vec->layer_size_);
  util::ReadBasicItem(in, &tag2vec->min_count_);
  util::ReadBasicItem(in, &tag2vec->sample_);
  util::ReadBasicItem(in, &tag2vec->init_alpha_);
  util::ReadBasicItem(in, &tag2vec->min_alpha_);
  tag2vec->Initialize();

  Vocabulary::Read<Word>(in, &tag2vec->word_vocab_);
  Vocabulary::Read<Tag>(in, &tag2vec->tag_vocab_);
  util::Huffman::Read(in, &tag2vec->word_huffman_);

  util::ReadMatrix(in, &tag2vec->tagi_);
  util::ReadMatrix(in, &tag2vec->wordo_);

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
  CHECK(min_alpha_ < init_alpha_) << "init_alpha should not be less than min_alpha.";
  random_ = new Random;
}

void Tag2Vec::DownSample(std::vector<const Vocabulary::Item*>* words) {
  size_t len = 0;
  for (size_t i = 0; i < words->size(); ++i) {
    float probability = ((Word*)(*words)[i])->probability();
    if (probability == 1 || probability >= random_->Sample()) {
      (*words)[len++] = (*words)[i];
    }
  }
  words->resize(len);
}

}  // namespace embedding
}  // namespace deeplearning
