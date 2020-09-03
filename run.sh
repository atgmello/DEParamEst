#!/bin/sh

export JULIA_NUM_THREADS=16
save_path='./data/results'
test_cases='[1,2,3,4,5,6,7,8,9]'
num_samples='[10]'
variance_multiplier='[0.1]'
julia ./src/calc/experiment.jl $save_path $test_cases $num_samples $variance_multiplier
