# Compiler (override on macOS: make CXX=g++-15)
CXX  = g++
NVCC = nvcc

# Host C++ flags
CXXFLAGS = -O3 -march=native -flto -std=c++17 -IInclude -static-libstdc++

# CUDA flags (override arch: make CUDA_ARCH=-arch=sm_80)
CUDA_ARCH ?= -arch=native
CUDAFLAGS = -O3 -std=c++17 $(CUDA_ARCH) -IInclude

# Shared sources
COMMON_SRC = Src/Main.cpp \
             Lib/Bucket_Partitioned_MDS/CVRP.cpp \
             Lib/Bucket_Partitioned_MDS/Solution.cpp \
             Lib/Command_Line_Args.cpp \
             Lib/Initializer.cpp \
             $(shell find Lib/Utils -name '*.cpp')

GPU_SRC = Lib/Gpu/DeviceData.cu \
          Lib/Gpu/BucketKernels.cu \
          Lib/Gpu/MstKernels.cu \
          Lib/Gpu/SolverContext.cu

COMMON_OBJ = $(COMMON_SRC:.cpp=.o)
GPU_OBJ    = $(GPU_SRC:.cu=.o)

# Main solver (default)
TARGET = Bin/bucket-partitioned-MDS

# Benchmarking / ablation binaries (CPU-only solvers, no CUDA)
TARGET_SET = Bin/bucket-partitioned-MDS-set
TARGET_DFS = Bin/bucket-partitioned-MDS-dfs
TARGET_BFS = Bin/bucket-partitioned-MDS-bfs
TARGET_BKT = Bin/bucket-partitioned-MDS-buckets

BENCH_TARGETS = $(TARGET_SET) $(TARGET_DFS) $(TARGET_BFS) $(TARGET_BKT)

# Default: main solver with CUDA
all: $(TARGET)

# All variants used for benchmarking (CPU-only)
bench-marking: $(BENCH_TARGETS)

$(TARGET): $(COMMON_OBJ) $(GPU_OBJ) Lib/Bucket_Partitioned_MDS/Solver.o
	@mkdir -p Bin
	$(CXX) $(CXXFLAGS) -flto $^ -o $@ -lcudart
	@echo "Build successful: $@"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

Lib/Bucket_Partitioned_MDS/Solver.o: Lib/Bucket_Partitioned_MDS/Solver.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_SET): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_cpp_set.cpp
	@mkdir -p Bin
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build successful: $@"

$(TARGET_DFS): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_Non_Lazy_DFS.cpp
	@mkdir -p Bin
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build successful: $@"

$(TARGET_BFS): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_BFS.cpp
	@mkdir -p Bin
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build successful: $@"

$(TARGET_BKT): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_buckets.cpp
	@mkdir -p Bin
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build successful: $@"

clean:
	rm -rf Bin/*
	rm -f $(COMMON_OBJ) $(GPU_OBJ) Lib/Bucket_Partitioned_MDS/Solver.o
	@echo "Cleaned build artifacts"

.PHONY: all bench-marking clean
