# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -O3
LDFLAGS ?=

NVCC ?= nvcc
NVCCFLAGS ?= -O3 --gpu-architecture compute_61

SOURCES := $(shell find src -name '*cpp')

SEQ_OBJECTS = $(SOURCES:src/%.cpp=build/seq-%.o)
OMP_OBJECTS = $(SOURCES:src/%.cpp=build/omp-%.o)

CUDA_SOURCES := $(shell find src -name '*cu')
CUDA_OBJECTS := $(CUDA_SOURCES:src/%.cu=build/cuda-%.o)

print-%: ; @echo $* = $($*)

omp cuda: CXXFLAGS += -fopenmp
omp cuda: LDFLAGS += -fopenmp

cuda: CXXFLAGS += -DCUDA -lcudart
cuda: LDFLAGS += -L/usr/local/depot/cuda-8.0/lib64/ -lcudart # this is different for Latedays

all: seq omp cuda

dirs:
	mkdir -p build

seq: dirs $(SEQ_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SEQ_OBJECTS) -o apsp-$@

omp: dirs $(OMP_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OMP_OBJECTS) -o apsp-$@

cuda: dirs $(OMP_OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OMP_OBJECTS) $(CUDA_OBJECTS) -o apsp-$@

build/seq-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/omp-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/cuda-%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) -r build

	$(RM) apsp-seq
	$(RM) apsp-omp
	$(RM) apsp-cuda

	$(RM) -r solution_cache
