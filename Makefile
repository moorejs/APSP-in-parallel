CXX = g++
OUTPUT = apsp

SRC_DIR = src
SRC_EXT = cpp
INCLUDE = -I$(SRC_PATH)

COMPILE_FLAGS = -std=c++11 -Wall -Wextra -g
RCOMPILE_FLAGS = -D NDEBUG
DCOMPILE_FLAGS = -D DEBUG

LINK_FLAGS = 

SOURCES := $(shell find $(SRC_DIR) -name '*$(SRC_EXT)')
OBJECTS = $(SOURCES:$(SRC_DIR)/%.$(SRC_EXT)=%.o)

release: export VERSION := release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS) $(RCOMPILE_FLAGS)
release: export LDFLAGS := $(LDFLAGS) $(LINK_FLAGS)

debug: export VERSION := debug
debug: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS) $(DCOMPILE_FLAGS)
debug: export LDFLAGS := $(LDFLAGS) $(LINK_FLAGS)

seq: export TYPE := seq

omp: export TYPE := omp
omp: export CXXFLAGS += -fopenmp
omp: export LD_FLAGS += -fopenmp

release debug: seq omp

seq omp:
	@mkdir -p build-seq
	@mkdir -p build-omp
	@mkdir -p bin/$(VERSION)
	$(MAKE) all --no-print-directory

all: bin/$(VERSION)/$(OUTPUT)
	@$(RM) $(OUTPUT)-seq
	@$(RM) $(OUTPUT)-omp
	@ln -s $<-seq $(OUTPUT)-seq
	@ln -s $<-omp $(OUTPUT)-omp

# linking
bin/$(VERSION)/$(OUTPUT): $(addprefix build-$(TYPE)/, $(OBJECTS))
	$(CXX) $^ -o $@-$(TYPE)

# compiling
build-$(TYPE)/%.o: $(SRC_DIR)/%.$(SRC_EXT)
	$(CXX) $(COMPILE_FLAGS) $(INCLUDE) -MP -MMD -c $< -o $@

clean:
	$(RM) $(OUTPUT)-seq
	$(RM) $(OUTPUT)-openmp

	$(RM) -r build-seq
	$(RM) -r build-omp

	$(RM) -r bin
	$(RM) -r solution_cache
