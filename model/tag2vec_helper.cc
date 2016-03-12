#include "model/tag2vec_helper.h"

#include <cmath>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/tag.h"
#include "document/vocabulary.h"
#include "document/word.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {

Tag2Vec::Random::Random() {
  std::random_device rd;
  engine_.seed(rd());
  sample_ = new std::uniform_real_distribution<float>(0, 1);
}

Tag2Vec::Random::~Random() {
  if (sample_) delete sample_;
}

float Tag2Vec::Random::Sample() {
  float ans = 0;
  #pragma omp critical(Tag2VecRandom)
  ans = (*sample_)(engine_);
  return ans;
}

void BuildWordVocabulary(DocumentIterator* iterator, size_t min_count,
                         float sample, Vocabulary* vocabulary) {
  const Document* document;
  while ((document = iterator->NextDocument())) {
    for (const std::string& word : document->words()) {
      vocabulary->AddItem<Word>(word);
    }
    delete document;
  }
  vocabulary->Build(min_count);

  if (sample < 0 || sample > 1) sample = 0;
  float threshold = sample * vocabulary->num_retained();
  for (size_t i = 0; i < vocabulary->items_size(); ++i) {
    Word* word = (Word*)vocabulary->item(i);
    float probability =
        (std::sqrt(word->count() / threshold) + 1) * (threshold / word->count());
    word->set_probability(std::min(probability, 1.0f));
  }
}

void BuildTagVocabulary(DocumentIterator* iterator, Vocabulary* vocabulary) {
  const Document* document;
  while ((document = iterator->NextDocument())) {
    for (const std::string& tag : document->tags()) {
      vocabulary->AddItem<Tag>(tag);
    }
  }
  vocabulary->Build();
}

void GetVocabularyItemVec(const Vocabulary& vocabulary,
                          const std::vector<std::string>& item_strs,
                          std::vector<const Vocabulary::Item*>* item_vec) {
  for (const std::string& item_str : item_strs) {
    const Vocabulary::Item* item = vocabulary.item(item_str);
    if (item) {
      item_vec->push_back(item);
    }
  }
}

}  // namespace embedding
}  // namespace deeplearning
