#!/bin/bash

# Randomly take specific percent lines of the input file
NUM = $(bc -l<<<"$1")
echo "$NUM"
cat $2 | awk 'BEGIN {srand()} !/^$/ { if (rand() <= $NUM) print $0}'
