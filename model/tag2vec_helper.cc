#include "model/tag2vec_helper.h"

#include <omp.h>
#include <random>

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

}  // namespace embedding
}  // namespace deeplearning
