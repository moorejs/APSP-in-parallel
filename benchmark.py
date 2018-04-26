#!/usr/bin/python

import subprocess
import re # regex
import argparse

bench_list = [
    [
        { 'n': 256, 't': 12, 'p': 0.25 },
        { 'n': 512, 't': 12, 'p': 0.25 },
        { 'n': 1024, 't': 12, 'p': 0.25 },
        { 'n': 1024+512, 't': 12, 'p': 0.25 },
    ],
    [
        { 'n': 256, 't': 12, 'p': 0.5 },
        { 'n': 512, 't': 12, 'p': 0.5 },
        { 'n': 1024, 't': 12, 'p': 0.5 },
        { 'n': 1024+512, 't': 12, 'p': 0.5 },
    ],
    [
        { 'n': 256, 't': 12, 'p': 0.75 },
        { 'n': 512, 't': 12, 'p': 0.75 },
        { 'n': 1024, 't': 12, 'p': 0.75 },
        { 'n': 1024+512, 't': 12, 'p': 0.75 },
    ],
    [
        { 'n': 256, 't': 12, 'p': 1.0 },
        { 'n': 512, 't': 12, 'p': 1.0 },
        { 'n': 1024, 't': 12, 'p': 1.0 },
        { 'n': 1024+512, 't': 12, 'p': 1.0 },
    ]
]

thread_bench = [
    [
        { 'n': 1024, 't': 1, 'p': 0.5 },
        { 'n': 1024, 't': 2, 'p': 0.5 },
        { 'n': 1024, 't': 3, 'p': 0.5 },
        { 'n': 1024, 't': 4, 'p': 0.5 },
        { 'n': 1024, 't': 5, 'p': 0.5 },
        { 'n': 1024, 't': 6, 'p': 0.5 },
        { 'n': 1024, 't': 7, 'p': 0.5 },
        { 'n': 1024, 't': 8, 'p': 0.5 },
        { 'n': 1024, 't': 9, 'p': 0.5 },
        { 'n': 1024, 't': 10, 'p': 0.5 },
        { 'n': 1024, 't': 11, 'p': 0.5 },
        { 'n': 1024, 't': 12, 'p': 0.5 },
    ]
]

BENCH_TO_USE = thread_bench

DEFAULT_BLOCK_SIZE = 8

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', choices=['f', 'j'], required=True,
                    help='Algorithm to benchmark')
parser.add_argument('-s', '--seed', default=42,
                    help='Seed for graph generation')
parser.add_argument('-d', '--block_size', default=DEFAULT_BLOCK_SIZE, 
                    help='The block size of the graph for Floyd-Warshall')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print commands as they run')
args = parser.parse_args()

def create_cmd(params):
    cmd = []
    for attr, value in params.iteritems():
        cmd += ['-' + attr, str(value)]
    
    return cmd

def run_cmd(command, verbose):
    if verbose:
        print 'Running command ' + ' '.join(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

def extract_time(stdout):
    return float(re.search(r'(\d*\.?\d*)ms', stdout).group(1))

def run_bench(bench_list, algorithm, seed, block_size, verbose):
    print ''
    print ' {:-^52} '.format('')
    print '|{:^52}|'.format('  Benchmark for {}\'s Algorithm  '
                             .format('Floyd-Warshall' if algorithm is 'f' else 'Johnson'))
    print '|{:^52}|'.format('seed = {}{}'.format(seed, ', block size = {}'.format(block_size) if algorithm is 'f' else ''))
    print ' {:-^52} '.format('')
    print '| {:<4} | {:<5} | {:<2} | {:<8} | {:<8} | {:<8} |'.format('p', 'n', 't', 'seq (ms)', 
                                                                     'par (ms)', 'speedup')

    for bench in bench_list:
        print ' {:-^52} '.format('')

        for param_obj in bench:
            param_obj['a'] = algorithm
            param_obj['s'] = seed
            params = create_cmd(param_obj)

            stdout, stderr = run_cmd(['./apsp-seq'] + params, verbose)

            if len(stderr):
                print 'Sequential Error: ' + stderr
                return
                
            seq_time = extract_time(stdout)
                
            stdout, stderr = run_cmd(['./apsp-omp'] + params, verbose)
                
            if len(stderr):
                print 'OMP Error: ' + stderr
                return

            omp_time = extract_time(stdout)

            print '| {p:>4.2f} | {n:>5} | {t:>2} | {:>8.1f} | {:>8.1f} | {:>7.1f}x |'.format(seq_time, omp_time, 
                                                                                             seq_time / omp_time, 
                                                                                             **param_obj)

    print ' {:-^52} '.format('')
    print ''
        
run_bench(BENCH_TO_USE, args.algorithm, args.seed, args.block_size, args.verbose)
