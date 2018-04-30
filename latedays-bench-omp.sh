#!/bin/bash

# The following modules must be loaded to compile (use .bashrc): 
# module load gcc-4.9.2
# module load binutils-2.26
# module load boost-1.61.0
# module load cuda-8.0

# Use 'qsub <script_name> -q timer' to submit job to Lateday cluster
PBS -lwalltime=0:45:00 # time limit
PBS -l nodes=1:ppn=24 # use 24 processors

cd $PBS_O_WORKDIR

# download argparse.py because python 2.6 doesn't have it
echo "Download argparse.py..."
curl -L https://github.com/ThomasWaldmann/argparse/raw/master/argparse.py > argparse.py

echo -e "\n\nMaking executables...\n"
LATEDAYS=1 make apsp-seq -j
LATEDAYS=1 make apsp-omp -j

echo -e "\n\nBenchmarking..."
./benchmark.py --algorithm f --benchmark serious
