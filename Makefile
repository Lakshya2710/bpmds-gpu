CXX  = g++
NVCC = nvcc

# Host C++ flags (no -flto: incompatible with nvcc final link on mixed toolchains)
CXXFLAGS = -O3 -march=native -std=c++17 -IInclude -static-libstdc++

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
          Lib/Gpu/RouteKernels.cu \
          Lib/Gpu/SolverContext.cu

COMMON_OBJ = $(COMMON_SRC:.cpp=.o)
GPU_OBJ    = $(GPU_SRC:.cu=.o)

# Main solver (default)
TARGET = Bin/bucket-partitioned-MDS

# Dynamic CVRP: pool rebuild with the same GPU solver
DYNAMIC_TARGET = Bin/dynamic-bpmds
UTIL_SRC = $(shell find Lib/Utils -name '*.cpp')
UTIL_OBJ = $(UTIL_SRC:.cpp=.o)
DYNAMIC_OBJ = Src/DynamicMain.o \
              Lib/Bucket_Partitioned_MDS/Dynamic.o \
              Lib/Gpu/NearestKernels.o \
              Lib/Gpu/InterRouteMoves.o

# Default: main solver with CUDA
all: $(TARGET) $(DYNAMIC_TARGET)

$(TARGET): $(COMMON_OBJ) $(GPU_OBJ) Lib/Bucket_Partitioned_MDS/Solver.o
	@mkdir -p Bin
	$(NVCC) $(CUDAFLAGS) -o $@ $^ -Xcompiler="$(CXXFLAGS)" -Xcompiler=-pthread
	@echo "Build successful: $@"

$(DYNAMIC_TARGET): $(DYNAMIC_OBJ) \
                   Lib/Bucket_Partitioned_MDS/CVRP.o \
                   Lib/Bucket_Partitioned_MDS/Solution.o \
                   Lib/Bucket_Partitioned_MDS/Solver.o \
                   $(GPU_OBJ) \
                   $(UTIL_OBJ)
	@mkdir -p Bin
	$(NVCC) $(CUDAFLAGS) -o $@ $^ -Xcompiler="$(CXXFLAGS)" -Xcompiler=-pthread
	@echo "Build successful: $@"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

Lib/Bucket_Partitioned_MDS/Solver.o: Lib/Bucket_Partitioned_MDS/Solver.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf Bin/*
	rm -f $(COMMON_OBJ) $(GPU_OBJ) $(DYNAMIC_OBJ) Lib/Bucket_Partitioned_MDS/Solver.o
	@echo "Cleaned build artifacts"

.PHONY: all clean
