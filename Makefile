# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -O3
LDFLAGS ?=

SOURCES := $(shell find src -name '*cpp')

omp: LDFLAGS += -fopenmp

all: seq omp

seq omp: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o apsp-$@

clean:
	$(RM) apsp-seq
	$(RM) apsp-omp
	$(RM) -r solution_cache
