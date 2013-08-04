#!/bin/bash

for i in 1 2 4 8 16 32 64 128 256
do
    echo $i
cat >nbody_weak_hpx_${i}.job<<EOF
#!/bin/bash
#SBATCH -J nbody_weak_hpx_${i}
#SBATCH -o nbody_weak_hpx_${i}.o%j
#SBATCH -e nbody_weak_hpx_${i}.e%j
#SBATCH -p normal
#SBATCH -N $i
#SBATCH -n $i
#SBATCH -t 01:00:00

APP="\$WORK/build/nbody_demo/release-host/bin/nbody_hpx_weak -Ihpx.parcel.mpi.enable=1 -Ihpx.parcel.tcpip.enable=0 -Ihpx.stacks.use_guard_pages=0"
PERFCTRS="--hpx:print-counter /agas{locality#*/total}/count/cache-evictions"
PERFCTRS="\$PERFCTRS --hpx:print-counter /agas{locality#*/total}/count/cache-hits"
PERFCTRS="\$PERFCTRS --hpx:print-counter /agas{locality#*/total}/count/cache-insertions"
PERFCTRS="\$PERFCTRS --hpx:print-counter /agas{locality#*/total}/count/cache-misses"
#PERFCTRS="\$PERFCTRS --hpx:print-counter-interval 20"

ibrun numactl --interleave=0,1 \$APP \$PERFCTRS

EOF

cat >nbody_weak_hpx_tcp_${i}.job<<EOF
#!/bin/bash
#SBATCH -J nbody_weak_hpx_tcp_${i}
#SBATCH -o nbody_weak_hpx_tcp_${i}.o%j
#SBATCH -e nbody_weak_hpx_tcp_${i}.e%j
#SBATCH -p normal
#SBATCH -N $i
#SBATCH -n $i
#SBATCH -t 01:00:00

HOSTS=\`scontrol show hostname\`
APP="\$WORK/build/nbody_demo/release-host/bin/nbody_hpx_weak --hpx:nodes \$HOSTS --hpx:endnodes -Ihpx.stacks.use_guard_pages=0"

ibrun -n 1 -o 0 numactl --interleave=0,1 \$APP --hpx:console --hpx:iftransform="s/^c/i" &
ibrun -n $(($i - 1)) -o 1 numactl --interleave=0,1 \$APP --hpx:worker --hpx:iftransform="s/^c/i" &
wait

EOF

cat >nbody_weak_mpi_${i}.job<<EOF
#!/bin/bash
#SBATCH -J nbody_weak_mpi_${i}
#SBATCH -o nbody_weak_mpi_${i}.o%j
#SBATCH -e nbody_weak_mpi_${i}.e%j
#SBATCH -p normal
#SBATCH -n $(($i*16))
#SBATCH -t 01:00:00

APP="\$WORK/build/nbody_demo/release-host/bin/nbody_mpi_weak"

ibrun \$APP

EOF
done

