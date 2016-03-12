#include <iostream>
#include <omp.h>
#include <unistd.h>

namespace {

int i = 0;

void F(int p) {
  int k;
  #pragma omp critical(A)
  {
    k = i + 1;
    sleep(1);
    i = k;
    std::cerr << "f(" << p << "): " << i << std::endl;
  }
}

void G(int p) {
  int k;
  #pragma omp critical(A)
  {
    k = i + 1;
    sleep(1);
    i = k;
    std::cerr << "g(" << p << "): " << i << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  omp_set_num_threads(4);
  #pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    if (i & 1) F(i);
    else G(i);
  }
  return 0;
}
