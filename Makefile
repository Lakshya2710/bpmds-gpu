# Compiler (override on macOS: make CXX=g++-15)
CXX  = g++
NVCC = nvcc

# CUDA toolkit path (on clusters: module load cuda, then make CUDA_HOME=$CUDA_ROOT)
CUDA_HOME ?= /usr/local/cuda
ifeq ($(wildcard $(CUDA_HOME)/lib64/libcudart.so),)
  ifeq ($(wildcard $(CUDA_HOME)/lib/libcudart.so),)
    CUDA_LIB_DIR =
  else
    CUDA_LIB_DIR = $(CUDA_HOME)/lib
  endif
else
  CUDA_LIB_DIR = $(CUDA_HOME)/lib64
endif

# Host C++ flags
CXXFLAGS = -O3 -march=native -std=c++17 -IInclude -static-libstdc++

# CUDA flags (override arch: make CUDA_ARCH=-arch=sm_80)
CUDA_ARCH ?= -arch=native
CUDAFLAGS = -O3 -std=c++17 $(CUDA_ARCH) -IInclude

# Link flags — explicit cudart path fixes "cannot find -lcudart" on PBS/cluster nodes
ifneq ($(CUDA_LIB_DIR),)
  CUDALINK = -L$(CUDA_LIB_DIR) -lcudart -Xcompiler=-pthread
else
  CUDALINK = -lcudart -Xcompiler=-pthread
endif

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

TARGET = Bin/bucket-partitioned-MDS

all: $(TARGET)

$(TARGET): $(COMMON_OBJ) $(GPU_OBJ) Lib/Bucket_Partitioned_MDS/Solver.o
	@mkdir -p Bin
	$(NVCC) $(CUDAFLAGS) -o $@ $^ -Xcompiler="$(CXXFLAGS)" $(CUDALINK)
	@echo "Build successful: $@"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

Lib/Bucket_Partitioned_MDS/Solver.o: Lib/Bucket_Partitioned_MDS/Solver.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf Bin/*
	rm -f $(COMMON_OBJ) $(GPU_OBJ) Lib/Bucket_Partitioned_MDS/Solver.o
	@echo "Cleaned build artifacts"

.PHONY: all clean
