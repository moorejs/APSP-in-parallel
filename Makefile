# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -O3
LDFLAGS ?=

NVCC ?= nvcc
NVCCFLAGS ?= -O3 --gpu-architecture compute_61

OBJ_DIR := objs

SEQ := apsp-seq
OMP := apsp-omp
CUDA := apsp-cuda

SOURCES := $(shell find src -name '*cpp')

SEQ_OBJECTS = $(SOURCES:src/%.cpp=$(OBJ_DIR)/seq-%.o)
OMP_OBJECTS = $(SOURCES:src/%.cpp=$(OBJ_DIR)/omp-%.o)

CUDA_SOURCES := $(shell find src -name '*cu')
CUDA_OBJECTS := $(CUDA_SOURCES:src/%.cu=$(OBJ_DIR)/cuda-%.o)

$(OMP) $(CUDA): CXXFLAGS += -fopenmp
$(OMP) $(CUDA): LDFLAGS += -fopenmp

$(CUDA): CXXFLAGS += -DCUDA -lcudart

ifeq ($(LATEDAYS),) # if LATEDAYS not set
$(CUDA): LDFLAGS += -L/usr/local/depot/cuda-8.0/lib64/ -lcudart
else
CXXFLAGS += -I/opt/boost-1.61.0/include -Isrc/
$(CUDA): LDFLAGS += -L/opt/cuda-8.0/lib64 -lcudart
endif

all: $(SEQ) $(OMP) $(CUDA)

$(SEQ): $(SEQ_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OMP): $(OMP_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(CUDA): $(OMP_OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/seq-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/omp-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/cuda-%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) -r $(OBJ_DIR)

	$(RM) $(SEQ)
	$(RM) $(OMP)
	$(RM) $(CUDA)

