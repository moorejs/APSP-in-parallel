# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -O3
CXXFLAGS += $(CXXEXTRA)
LDFLAGS ?=

ISPC ?= ispc
ISPC_FLAGS ?= --arch=x86-64 --emit-obj -g -03

NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O3

OBJ_DIR := objs
SRC_DIR := src

# executable names
SEQ := apsp-seq
OMP := apsp-omp
CUDA := apsp-cuda
SEQ_ISPC = $(SEQ)-ispc
OMP_ISPC = $(OMP)-ispc

CPP_SOURCES := $(shell find $(SRC_DIR) -name '*cpp')
CU_SOURCES := $(shell find $(SRC_DIR) -name '*cu')
ISPC_SOURCES := $(shell find $(SRC_DIR) -name '*ispc')

SEQ_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/seq-%.o)
OMP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/omp-%.o)

CUDA_CPP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/cuda-cpp-%.o)
CUDA_CU_OBJECTS := $(CU_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/cuda-cu-%.o)

ISPC_SEQ_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/ispc-seq-%.o)
ISPC_OMP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/ispc-omp-%.o)
ISPC_OBJECTS := $(ISPC_SOURCES:$(SRC_DIR)/%.ispc=$(OBJ_DIR)/ispc-%.o)

$(OMP) $(OMP_ISPC) $(CUDA): CXXFLAGS += -fopenmp
$(OMP) $(OMP_ISPC) $(CUDA): LDFLAGS += -fopenmp

$(CUDA): CXXFLAGS += -DCUDA -lcudart

$(SEQ_ISPC) $(OMP_ISPC): CXXFLAGS += -DISPC

ifeq ($(LATEDAYS),) # if LATEDAYS not set, assume GHC machines
$(SEQ_ISPC) $(OMP_ISPC): ISPCFLAGS += --target=avx2-i32x8
$(CUDA): LDFLAGS += -L/usr/local/depot/cuda-8.0/lib64/ -lcudart
$(CUDA): NVCCFLAGS += -arch=compute_61 -code=sm_61
else
CXXFLAGS += -I/opt/boost-1.61.0/include -I$(SRC_DIR)/
$(CUDA): LDFLAGS += -L/opt/cuda-8.0/lib64 -lcudart
$(CUDA): NVCCFLAGS += -arch=compute_35 -code=sm_35
endif

all: $(SEQ) $(OMP) $(CUDA) $(SEQ_ISPC) $(OMP_ISPC)

$(OBJ_DIR):
	mkdir -p $@

$(SEQ): $(SEQ_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OMP): $(OMP_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(CUDA): $(CUDA_CPP_OBJECTS) $(CUDA_CU_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(SEQ_ISPC): $(ISPC_SEQ_OBJECTS) $(ISPC_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OMP_ISPC): $(ISPC_OMP_OBJECTS) $(ISPC_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

-include $(SEQ_OBJECTS:%.o=%.d)
-include $(OMP_OBJECTS:%.o=%.d)
-include $(CUDA_CPP_OBJECTS:%.o=%.d)

$(OBJ_DIR)/seq-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/omp-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/cuda-cpp-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/cuda-cu-%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/ispc-seq-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/ispc-omp-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/ispc-%.o: $(SRC_DIR)/%.ispc | $(OBJ_DIR)
	$(ISPC) $(ISPCFLAGS) $< -o $@
# we do not output a header here on purpose

clean:
	$(RM) -r $(OBJ_DIR)
	$(RM) $(SEQ) $(OMP) $(CUDA) $(SEQ_ISPC) $(OMP_ISPC)

