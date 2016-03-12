#ifndef MODEL_TAG2VEC_HELPER_H_
#define MODEL_TAG2VEC_HELPER_H_

#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/vocabulary.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {

class Tag2Vec::Random final {
 public:
  Random();
  Random(const Random& random) = delete;
  ~Random();

  std::mt19937_64& engine() { return engine_; }

  float Sample();

 private:
  std::mt19937_64 engine_;
  std::uniform_real_distribution<float>* sample_ = nullptr;  // OWNED
};

void BuildWordVocabulary(DocumentIterator* iterator, size_t min_count,
                         float sample, Vocabulary* vocabulary);

void BuildTagVocabulary(DocumentIterator* iterator, Vocabulary* vocabulary);

void GetVocabularyItemVec(const Vocabulary& vocabulary,
                          const std::vector<std::string>& item_strs,
                          std::vector<const Vocabulary::Item*>* item_vec);

//void TrainSgPair(Tag2Vec::RMatrixXf* tagi, Tag2Vec::RMatrixXf* wordo);

}  // namespace embedding
}  // namespace deeplearning

#endif
