#ifndef UTIL_LOGGING_H_
#define UTIL_LOGGING_H_

// Advanced logging system.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#define DEBUG_MESSAGE __FILE__ << ":" << __LINE__ << ":" << __func__ << ": "

#define CHECK_EQ(expect, actual)                                             \
  search::util::CheckEqualMessage<decltype(actual)>(expect, actual).stream() \
      << DEBUG_MESSAGE

#define CHECK_EQ_DEBUG(expect, actual)                                   \
  search::util::CheckEqualDebugMessage<decltype(actual)>(expect, actual) \
          .stream()                                                      \
      << DEBUG_MESSAGE

#define CHECK_EQ_FLOAT(expect, actual, eps)                          \
  search::util::CheckEqualFloatMessage(expect, actual, eps).stream() \
      << DEBUG_MESSAGE

#define CHECK(condition) \
  search::util::CheckConditionMessage(condition).stream() << DEBUG_MESSAGE

#define LOG(severity) LOG_##severity.stream() << DEBUG_MESSAGE
#define LOG_INFO search::util::LogInfoMessage()
#define LOG_FATAL search::util::CheckConditionMessage(true)

namespace search {
namespace util {

class Logger {
 public:
  std::ostream& stream() { return ostr_; }

 protected:
  std::ostringstream ostr_;
};

template <typename T>
class CheckEqualMessage : public Logger {
 public:
  CheckEqualMessage(const T& expect, const T& actual)
      : expect_(expect), actual_(actual) {}

  ~CheckEqualMessage() {
    if (expect_ == actual_) return;

    std::cerr << ostr_.str() << std::endl;
    std::cerr << "Expect: " << expect_ << std::endl;
    std::cerr << "Actual: " << actual_ << std::endl;
    exit(1);
  }

 private:
  T expect_;
  T actual_;
};

template <typename T>
class CheckEqualDebugMessage : public Logger {
 public:
  CheckEqualDebugMessage(const T& expect, const T& actual)
      : expect_(expect), actual_(actual) {}

  ~CheckEqualDebugMessage() {
    if (expect_ == actual_) return;

    std::cerr << ostr_.str() << std::endl;
    std::cerr << "Expect: " << expect_.DebugString() << std::endl;
    std::cerr << "Actual: " << actual_.DebugString() << std::endl;
    exit(1);
  }

 private:
  T expect_;
  T actual_;
};

class CheckEqualFloatMessage : public Logger {
 public:
  CheckEqualFloatMessage(float expect, float actual, float eps)
      : expect_(expect), actual_(actual), eps_(eps) {}

  ~CheckEqualFloatMessage() {
    if (std::abs(expect_ - actual_) < eps_) return;

    std::cerr << ostr_.str() << std::endl;
    std::cerr << "Expect: " << expect_ << std::endl;
    std::cerr << "Actual: " << actual_ << std::endl;
    std::cerr << "Delta = " << std::abs(expect_ - actual_) << " >= " << eps_
              << std::endl;
    exit(1);
  }

 private:
  float expect_;
  float actual_;
  float eps_;
};

template <typename T>
class CheckEqualMessageNull : public Logger {
 public:
  CheckEqualMessageNull(const T& expect, const T& actual)
      : expect_(expect), actual_(actual) {}

  ~CheckEqualMessageNull() {
    if (expect_ == actual_) return;

    std::cerr << ostr_.str() << std::endl;
    exit(1);
  }

 private:
  T expect_;
  T actual_;
};

class CheckConditionMessage : public Logger {
 public:
  CheckConditionMessage(bool condition) : condition_(condition) {}

  ~CheckConditionMessage() {
    if (condition_) return;

    std::cerr << ostr_.str() << std::endl;
    exit(1);
  }

 private:
  bool condition_;
};

class LogInfoMessage : public Logger {
 public:
  ~LogInfoMessage() { std::cout << ostr_.str() << std::endl; }
};

}  // namespace util
}  // namespace search

#endif  // UTIL_LOGGING_H_
