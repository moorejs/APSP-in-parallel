# APSP-in-parallel

Implementation of Floyd-Warshall's Algorithm and Johnson's Algorithm using different methods of expressing parallelism. We sought to take strong performing sequential versions of both algorithms and compare them to the their parallel counterparts.

Our full project description and results can be found here: https://moorejs.github.io/APSP-in-parallel/

Floyd-Warshall has its sequential version, OpenMP, CUDA and ISPC while Johnson's Algorithm has its sequential version, OpenMP and CUDA.

For Johnson's Algorithm we require the Boost Graph Library as we use it to create our baseline sequential implementation.

The sequential versions of the Floyd-Warshall and Johnson can be further optimized using PGO in profile.py. Follow the instructions on our site to achieve guided optimization speedup for the sequential version.

Once you compile the code, you can run different benchmarks using benchmark.py. You can use ./benchmark.py -h to see the different parameters
