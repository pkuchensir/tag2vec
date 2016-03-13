#ifndef MODEL_MODEL_H_
#define MODEL_MODEL_H_

#include <ostream>
#include <vector>

#include "document/document.h"
#include "document/score_item.h"

namespace deeplearning {
namespace embedding {

class Model {
 public:
  virtual ~Model() = default;

  virtual void Train(DocumentIterator* iterator, size_t iter) = 0;
  virtual std::vector<ScoreItem> Suggest(
      const std::vector<std::string>& words) = 0;
  virtual void Write(std::ostream* out) const = 0;
};

}  // namespace embedding
}  // namespace deeplearning

#endif
