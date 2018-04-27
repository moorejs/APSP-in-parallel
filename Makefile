# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -O3
LDFLAGS ?=

NVCC ?= nvcc
NVCCFLAGS ?= -O3 --gpu-architecture compute_61

SOURCES := $(shell find src -name '*cpp')
OBJECTS := $(SOURCES:src/%.cpp=build/%.o)

CUDA_SOURCES := $(shell find src -name '*cu')
CUDA_OBJECTS := $(CUDA_SOURCES:src/%.cu=build/cuda-%.o)

print-%: ; @echo $* = $($*)

omp: CXXFLAGS += -fopenmp
omp: LDFLAGS += -fopenmp

cuda: CXXFLAGS += -DCUDA -lcudart
cuda: LDFLAGS += -L/usr/local/depot/cuda-8.0/lib64/ -lcudart # this is different for Latedays

all: seq omp cuda

dirs:
	mkdir -p build

# compile and link in one step
seq omp: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o apsp-$@

# compile seperately (two compilers) then link
cuda: dirs $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o apsp-$@

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/cuda-%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) -r build

	$(RM) apsp-seq
	$(RM) apsp-omp
	$(RM) apsp-cuda

	$(RM) -r solution_cache
