#include <unistd.h>	// getopt
#include <chrono>		 // high_resolution_clock
#include <iostream>	// cout
#include <random>		 // mt19937_64
#include <ratio>		 // milli

void print_usage();

int main(int argc, char* argv[]) {
	unsigned long seed;
	double p;
	bool use_floyd_warshall = true;

	int opt;
	while ((opt = getopt(argc, argv, "ha:p:s:")) != -1) {
		switch (opt) {
			case 'h':
				print_usage();
				return 0;

			case 'a':
				if (optarg[0] == 'j') {
					use_floyd_warshall = false;
				} else if (optarg[0] != 'f') {
					return -1;	// illegal argument, must be f or j
				}
				break;

			case 'p':
				p = std::stod(optarg);
				break;

			case 's':
				seed = std::stoul(optarg);
				break;
		}
	}

	std::mt19937_64 rand_engine(seed);
	std::uniform_real_distribution<double> flip(0, 1);

	std::cout << "Two coin flips (based on seed if set): " << flip(rand_engine) << " " << flip(rand_engine) << "\n";
	// we can use this coin flipping technology to build or graphs
	// if (use_floyd_warshall) build adj_matrix
	// else build adj_list

	auto start = std::chrono::high_resolution_clock::now();

	int sum = 0;
	for (long i = 0; i < 1000000000L; i++) {
		sum += i;
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> start_to_end = end - start;
	std::cout << "\nTotal runtime: " << start_to_end.count() << "ms\n\n";

	return 0;
}

void print_usage() {
	std::cout << "\nUsage: asap [-a (f|j)] [-s seed]\n";
	std::cout << "\t-h\tPrint this message\n";
	std::cout << "\t-a\tAlgorithm to use for all pairs shortest path\n";
	std::cout << "\t\t\tf: Floyd-Warshall (default)\n";
	std::cout << "\t\t\tj: Johnson's Algorithm\n";
	std::cout << "\t-s\tSeed for graph generation\n";
	std::cout << "\n";
}