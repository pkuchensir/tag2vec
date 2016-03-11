#ifndef DOCUMENT_VOCABULARY_H_
#define DOCUMENT_VOCABULARY_H_

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace deeplearning {
namespace embedding {

class Vocabulary final {
 public:
  class Item;

 public:
  Vocabulary() = default;
  Vocabulary(const Vocabulary& vocabulary) = delete;
  ~Vocabulary();

  Item* item(size_t index) { return items_[index]; }
  const Item* item(size_t index) const { return items_[index]; }
  const Item* item(const std::string& item) const {
    auto it = item_hash_.find(item);
    return it != item_hash_.end() ? it->second : nullptr;
  }
  const size_t total_items() const { return total_items_; }
  const size_t items_size() const { return items_.size(); }

  template <class ItemSubClass>
  void AddItem(const std::string& item);

  void Build(size_t min_count);

  void Write(std::ostream* out) const;

  template <class ItemSubClass>
  static void Read(std::istream* in, Vocabulary* vocabulary);

  void Clear();

 private:
  bool CheckEmpty() const;

 private:
  size_t total_items_ = 0;
  std::vector<Item*> items_;
  std::unordered_map<std::string, Item*> item_hash_;

  bool has_built = false;
};

class Vocabulary::Item {
 public:
  Item() = default;
  Item(const std::string& text) : index_(0), count_(1), text_(text) {}
  Item(std::string&& text) : index_(0), count_(1), text_(std::move(text)) {}
  Item(size_t index, size_t count, const std::string& text)
      : index_(index), count_(count), text_(text) {}
  Item(size_t index, size_t count, std::string&& text)
      : index_(index), count_(count), text_(std::move(text)) {}

  virtual ~Item() = default;

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

#include "document/vocabulary.incl.h"

#endif  // DOCUMENT_VOCABULARY_H_
