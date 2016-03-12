#ifndef MODEL_TAG2VEC_HELPER_H_
#define MODEL_TAG2VEC_HELPER_H_

#include <eigen3/Eigen/Core>
#include <random>

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
  std::uniform_real_distribution<float>* sample_ = nullptr;
};

//void TrainSgPair(Tag2Vec::RMatrixXf* tagi, Tag2Vec::RMatrixXf* wordo);

}  // namespace embedding
}  // namespace deeplearning

#endif
