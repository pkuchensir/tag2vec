#ifndef MODEL_TAG2VEC_H_
#define MODEL_TAG2VEC_H_

#include <eigen3/Eigen/Core>
#include <iostream>
#include <string>

#include "document/document.h"
#include "document/tag.h"
#include "document/vocabulary.h"
#include "document/word.h"
#include "util/huffman.h"

namespace deeplearning {
namespace embedding {

class Tag2Vec final {
 private:
  class Random;

 public:
  using RMatrixXf =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

 public:
  Tag2Vec();
  Tag2Vec(size_t layer_size, size_t min_count, float sample, float init_alpha,
          float min_alpha);
  Tag2Vec(const Tag2Vec& tag2vec) = delete;
  ~Tag2Vec();

  void Train(const std::vector<Document>& documents, size_t iter);
  void Train(DocumentIterator* iterator, size_t iter);

  // Eigen::VectorXf TagVec(const std::string& tag) const;
  // void MostSimilar(const Eigen::VectorXf& v) const;

  Eigen::RowVectorXf Infer(const std::vector<std::string>& words,
                           size_t iter) const;

  void Write(std::ostream* out) const;
  static void Read(std::istream* in, Tag2Vec* tag2vec);

  std::string ConfigString() const;

 private:
  void Initialize();

  void DownSample(std::vector<const Vocabulary::Item*>* words);

 private:
  size_t layer_size_ = 300;
  size_t min_count_ = 0;
  float sample_ = 0;
  float init_alpha_ = 0.025;
  float min_alpha_ = 0.0001;

  bool has_trained_ = false;

  Random* random_ = nullptr;  // OWNED

  Vocabulary word_vocab_, tag_vocab_;
  util::Huffman word_huffman_;

  RMatrixXf tagi_, wordo_;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // MODEL_TAG2VEC_H_
