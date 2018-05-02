# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -O3
LDFLAGS ?=

NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O3
# more NVCC flags added below depending on machine

OBJ_DIR := objs

SEQ := apsp-seq
OMP := apsp-omp
CUDA := apsp-cuda

SOURCES := $(shell find src -name '*cpp')

SEQ_OBJECTS := $(SOURCES:src/%.cpp=$(OBJ_DIR)/seq-%.o)
OMP_OBJECTS := $(SOURCES:src/%.cpp=$(OBJ_DIR)/omp-%.o)
CUDA_CPP_OBJECTS := $(SOURCES:src/%.cpp=$(OBJ_DIR)/cuda-cpp-%.o)

CUDA_CU_SOURCES := $(shell find src -name '*cu')
CUDA_CU_OBJECTS := $(CUDA_CU_SOURCES:src/%.cu=$(OBJ_DIR)/cuda-cu-%.o)

$(OMP) $(CUDA): CXXFLAGS += -fopenmp
$(OMP) $(CUDA): LDFLAGS += -fopenmp

$(CUDA): CXXFLAGS += -DCUDA -lcudart

ifeq ($(LATEDAYS),) # if LATEDAYS not set, assume GHC machines
$(CUDA): LDFLAGS += -L/usr/local/depot/cuda-8.0/lib64/ -lcudart
$(CUDA): NVCCFLAGS += -arch=compute_61 -code=sm_61
else
CXXFLAGS += -I/opt/boost-1.61.0/include -Isrc/
$(CUDA): LDFLAGS += -L/opt/cuda-8.0/lib64 -lcudart
$(CUDA): NVCCFLAGS += -arch=compute_35 -code=sm_35
endif

all: $(SEQ) $(OMP) $(CUDA)

$(SEQ): $(SEQ_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OMP): $(OMP_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(CUDA): $(CUDA_CPP_OBJECTS) $(CUDA_CU_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

-include $(SEQ_OBJECTS:%.o=%.d)
-include $(OMP_OBJECTS:%.o=%.d)
-include $(CUDA_CPP_OBJECTS:%.o=%.d)

$(OBJ_DIR)/seq-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/omp-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/cuda-cpp-%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/cuda-cu-%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) -r $(OBJ_DIR)

	$(RM) $(SEQ)
	$(RM) $(OMP)
	$(RM) $(CUDA)

