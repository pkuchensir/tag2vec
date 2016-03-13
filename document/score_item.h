#ifndef DOCUMENT_SCORE_ITEM_H_
#define DOCUMENT_SCORE_ITEM_H_

#include <string>
#include <utility>

namespace deeplearning {
namespace embedding {

class ScoreItem final {
 public:
  ScoreItem(const std::string& item, float score)
      : item_(item), score_(score) {}
  ScoreItem(std::string&& item, float score)
      : item_(std::move(item)), score_(score) {}

  bool operator<(const ScoreItem& score_item) const;
  bool operator>(const ScoreItem& score_item) const;
  bool operator==(const ScoreItem& score_item) const;
  bool operator!=(const ScoreItem& score_item) const;

  const std::string& item() const { return item_; }
  float score() const { return score_; }

  std::string ToString() const;

 private:
  std::string item_;
  float score_;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_SCORE_ITEM_H_
