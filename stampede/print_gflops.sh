#!/bin/bash

for i in 1 2 4 8 16 32 64 128 256
do
    for j in `ls $1/nbody_weak_*_$i.o*`
    do
        MODE=`echo $j | sed -n -e 's/.*_\([a-z]*\)_[0-9].*/\1/p'`
        echo -n "$MODE $i "
        grep GFLOP $j
        if [ x"$?" != x"0" ]
        then
            echo
        fi
    done
done
