#pragma once

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
int* floyd_warshall_init(const int n, const double p, const unsigned long seed);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
int* floyd_warshall_blocked_init(const int n, const int block_size, const double p, const unsigned long seed);

// expects len(input) == len(output) == n*n
void floyd_warshall(const int* input, int* output, const int n);

// expects len(input) == len(output) == n*n
void floyd_warshall_blocked(const int* input, int* output, const int n, const int b);

void floyd_warshall_blocked_cuda();
