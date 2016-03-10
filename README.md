# tag2vec
This project aims to build tag representation models based on word2vec.

---
## Platform
* Ubuntu 14.04
* g++ >= 4.8
* eigen3

---
## Build & Clean
```bash
# Build
make

# Clean
make clean
```

---
## Run all unit tests
```bash
# Normal Run
for i in `find -name "*_test"`; do $i; done

# Run with Heapcheck
for i in `find -name "*_test"`; do valgrind --tool=memcheck $i; done
```

---
