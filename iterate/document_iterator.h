#include <vector>
#include <string>

class Vocabuary {
 public:
  class Word {
  };

};

class TaggedDocument {
 public:
  std::vector<std::string> words;
  std::vector<std::string> tags;
};

class TaggedDocumentIterator {
 public:
  virtual ~TaggedDocumentIterator() = default;

  virtual bool HasNext() = 0;

  virtual TaggedDocument* NextDocument() = 0;
};


class Corpus {
 public:
  virtual ~Corpus() = default;

  virtual TaggedDocumentIterator* HasNext() = 0;
};

