#ifndef DOCUMENT_WORD_H_
#define DOCUMENT_WORD_H_

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace deeplearning {
namespace embedding {

class Vocabulary {
 public:
  class Item;

 public:
  const Item* item(size_t index) const { return items_[index]; }
  const Item* item(const std::string& item) const {
    auto it = item_hash_.find(item);
    return it != item_hash_.end() ? it->second : nullptr;
  }

  void Write(std::ostream* out) const;
  static void Read(std::istream* in, Vocabulary* item);

 private:
  std::vector<Item*> items_;
  std::unordered_map<std::string, Item*> item_hash_;
};

class Vocabulary::Item {
 public:
  Item(size_t index, size_t count, const std::string& text)
      : index_(index), count_(count), text_(text) {}
  Item(size_t index, size_t count, std::string&& text)
      : index_(index), count_(count), text_(std::move(text)) {}

  size_t index() const { return index_; }
  void set_index(size_t index) { index_ = index; }

  size_t count() const { return count_; }
  void set_count(size_t count) { count_ = count; }

  const std::string& text() const { return text_; }
  void set_text(const std::string& text) { text_ = text; }
  void set_text(std::string&& text) { text_ = std::move(text); }

  void Write(std::ostream* out) const;
  static void Read(std::istream* in, Item* item);

 private:
  size_t index_;
  size_t count_;
  std::string text_;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_WORD_H_
