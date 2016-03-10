#include "util/util.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {
namespace {

mode_t GetMode(const char* path) {
  struct stat st;
  int res = stat(path, &st);
  CHECK_EQ(0, res) << "Something wrong with " << path << ".";
  return st.st_mode;
}

bool IsDir(const char* path) { return S_ISDIR(GetMode(path)); }

bool IsFile(const char* path) { return S_ISREG(GetMode(path)); }

}  // namespace

std::string BasePath(const std::string& file_path) {
  size_t pos = file_path.find_last_of('/');
  if (pos == std::string::npos) return "";

  return file_path.substr(0, pos + 1);
}

std::string Filename(const std::string& file_path) {
  size_t pos = file_path.find_last_of('/');
  if (pos == std::string::npos) return file_path;

  return file_path.substr(pos + 1);
}

std::string JoinPath(const std::string& path, const std::string& filename) {
  return path.empty() || path.back() == '/' ? path + filename
                                            : path + "/" + filename;
}

std::string ReadContent(const std::string& path) {
  std::string ans;
  std::ifstream fin(path);
  fin.peek();
  while (!fin.eof()) {
    ans += fin.get();
    fin.peek();
  }
  fin.close();
  return ans;
}

std::vector<std::string> Split(const std::string& str, const char* delim) {
  std::vector<bool> delim_set(128);
  while (*delim) {
    delim_set[*delim] = true;
    ++delim;
  }
  std::vector<std::string> ans;
  size_t i = 0;
  while (i < str.size()) {
    if (delim_set[str[i]]) {
      ++i;
      continue;
    }
    size_t j = i + 1;
    while (j < str.size() && !delim_set[str[j]]) ++j;
    ans.push_back(str.substr(i, j - i));
    i = j;
  }
  return ans;
}

void System(const std::string& command) {
  int ans = system(command.c_str());
  CHECK_EQ(0, ans) << "Command failed: " << command;
}

void TraversePath(const std::string& path,
                  const std::function<void(const std::string&)>& func) {
  DIR* dir = opendir(path.c_str());
  struct dirent* ent;
  while ((ent = readdir(dir))) {
    std::string filename = ent->d_name;
    if (filename.find(".") == 0) continue;
    std::string full_path = JoinPath(path, filename);
    if (IsFile(full_path.c_str())) {
      func(full_path);
    } else if (IsDir(full_path.c_str())) {
      TraversePath(full_path, func);
    }
  }
  closedir(dir);
}

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning
