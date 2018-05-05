#!/usr/bin/python

import subprocess
import re # regex
import argparse


all_benchmarks = {
    'normal': [
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
    ],
    'brief': [
        [
            { 'n': 1024, 't': 12, 'p': 0.5 },
            { 'n': 2048, 't': 12, 'p': 0.5 },
        ]
    ],
    'thread_scale': [
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
    ],
    'serious': [
        [
            { 'n': 1024, 't': 12, 'p': 0.5 },
            { 'n': 2048, 't': 12, 'p': 0.5 },
            { 'n': 4096, 't': 12, 'p': 0.5 },
            { 'n': 8192, 't': 12, 'p': 0.5 },
        ]
    ]
}

DEFAULT_BENCH = 'normal'
DEFAULT_BLOCK_SIZE = 8

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', choices=['f', 'j'], required=True,
                    help='Algorithm to benchmark')
parser.add_argument('-s', '--seed', default=42,
                    help='Seed for graph generation')
parser.add_argument('-d', '--block_size', default=DEFAULT_BLOCK_SIZE,
                    help='The block size of the graph for Floyd-Warshall')
parser.add_argument('-b', '--benchmark', choices=all_benchmarks.keys(), default=DEFAULT_BENCH,
                    help='The name of the benchmark to run')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print commands as they run')
parser.add_argument('-g', '--cuda', action='store_true', help='Run CUDA version')
parser.add_argument('-r', '--compare', action='store_true', help='Compare different parallel schemes. Recommended to be used with "-b serious"')

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

def run_bench(bench_list, algorithm, seed, block_size, verbose, cuda, caching_seq=True, seq_cache={}):
    
    print ''
    print ' {0:-^52} '.format('')
    print '|{0:^52}|'.format('  Benchmark for {0}\'s Algorithm  '
                             .format('Floyd-Warshall' if algorithm is 'f' else 'Johnson'))
    print '|{0:^52}|'.format('seed = {0}{1}'.format(seed, ', block size = {0}'.format(block_size) if algorithm is 'f' else ''))
    print ' {0:-^52} '.format('')
    print '| {0:<4} | {1:<5} | {2:<2} | {3:<8} | {4:<8} | {5:<8} |'.format('p', 'n', 't', 'seq (ms)',
                                                                     'par (ms)', 'speedup')

    for bench in bench_list:
        print ' {0:-^52} '.format('')

        for param_obj in bench:
            param_obj['a'] = algorithm
            param_obj['s'] = seed
            param_obj['d'] = block_size
            params = create_cmd(param_obj)

            cache_key = str(param_obj['p']) + str(param_obj['n'])
            if not caching_seq or cache_key not in seq_cache:
                stdout, stderr = run_cmd(['./apsp-seq'] + params, verbose)

                if len(stderr):
                    print 'Sequential Error: ' + stderr
                    return

                seq_cache[cache_key] = extract_time(stdout)

            seq_time = seq_cache[cache_key]

            if cuda: stdout, stderr = run_cmd(['./apsp-cuda'] + params,verbose)
            else: stdout, stderr = run_cmd(['./apsp-omp'] + params, verbose)

            if len(stderr):
                print 'Parallel Error: ' + stderr
                return

            par_time = extract_time(stdout)

            print '| {p:>4.2f} | {n:>5} | {t:>2} | {0:>8.1f} | {1:>8.1f} | {2:>7.1f}x |'.format(seq_time, par_time,
                                                                                             seq_time / par_time,
                                                                                             **param_obj)

    print ' {0:-^52} '.format('')
    print ''

def run_par_bench(bench_list, algorithm, seed, block_size, verbose, caching_seq=True, seq_cache={}):
    
    print ''
    print ' {0:-^54} '.format('')
    print '|{0:^54}|'.format('  Benchmark for {0}\'s Algorithm  '
                             .format('Floyd-Warshall' if algorithm is 'f' else 'Johnson'))
    print '|{0:^54}|'.format('seed = {0}{1}'.format(seed, ', block size = {0}'.format(block_size) if algorithm is 'f' else ''))
    print ' {0:-^54} '.format('')
    print '| {0:<4} | {1:<5} | {2:<2} | {3:<8} | {4:<8} | {5:<8} |'.format('p', 'n', 't', 'ISPC (ms)',
                                                                     'OMP (ms)', 'CUDA (ms)')

    for bench in bench_list:
        print ' {0:-^54} '.format('')

        for param_obj in bench:
            param_obj['a'] = algorithm
            param_obj['s'] = seed
            param_obj['d'] = block_size
            params = create_cmd(param_obj)

            stdout, stderr = run_cmd(['./apsp-omp'] + params, verbose)
            if len(stderr):
                print 'OMP Error: ' + stderr
                return
            omp_time = extract_time(stdout)

            stdout, stderr = run_cmd(['./apsp-omp-ispc'] + params, verbose)
            if len(stderr):
                print 'OMP ISPC Error: ' + stderr
                return
            omp_ispc_time = extract_time(stdout)
            
            stdout, stderr = run_cmd(['./apsp-cuda'] + params,verbose)
            if len(stderr):
                print 'CUDA Error: ' + stderr
                return

            cuda_time = extract_time(stdout)

            print '| {p:>4.2f} | {n:>5} | {t:>2} | {0:>9.1f} | {1:>8.1f} | {2:>9.1f} |'.format(omp_ispc_time, omp_time,
                                                                                             cuda_time,
                                                                                             **param_obj)

    print ' {0:-^52} '.format('')
    print ''


def choose_benchmark():
    if (args.compare):
        run_par_bench(all_benchmarks[args.benchmark], args.algorithm, args.seed, args.block_size, args.verbose)
    else:
        run_bench(all_benchmarks[args.benchmark], args.algorithm, args.seed, args.block_size, args.verbose, args.cuda)

choose_benchmark()
