#include "document/score_item.h"

#include <string>

namespace deeplearning {
namespace embedding {

bool ScoreItem::operator<(const ScoreItem& score_item) const {
  if (score_ < score_item.score_) return true;
  if (score_ > score_item.score_) return false;
  return item_ < score_item.item_;
}

bool ScoreItem::operator>(const ScoreItem& score_item) const {
  return !(*this < score_item);
}

bool ScoreItem::operator==(const ScoreItem& score_item) const {
  return !(*this < score_item) && !(score_item < *this);
}

bool ScoreItem::operator!=(const ScoreItem& score_item) const {
  return *this < score_item || score_item < *this;
}

std::string ScoreItem::ToString() const {
  return item_ + ":" + std::to_string(score_);
}

}  // namespace embedding
}  // namespace deeplearning
