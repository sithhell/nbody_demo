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

APP_HOST="$WORK/nbody_demo/stampede/run_nbody_hpx_weak_host.sh"
ibrun \$APP_HOST

EOF

cat >nbody_weak_hpx_symm_${i}.job<<EOF
#!/bin/bash
#SBATCH -J nbody_weak_hpx_symm_${i}
#SBATCH -o nbody_weak_hpx_symm_${i}.o%j
#SBATCH -e nbody_weak_hpx_symm_${i}.e%j
#SBATCH -p normal-mic
#SBATCH -N $i
#SBATCH -n $i
#SBATCH -t 01:00:00

export DAPL_UCM_REP_TIME=8000
export DAPL_UCM_RTU_TIME=4000
export DAPL_UCM_RETRY=10

APP_HOST="$WORK/nbody_demo/stampede/run_nbody_hpx_weak_host.sh"
APP_MIC="$WORK/nbody_demo/stampede/run_nbody_hpx_weak_mic.sh"
#export SCALE_PROCS=$(($i *16 + $i * 60))
export MIC_PPN=1
ibrun.symm -c \$APP_HOST -m \$APP_MIC

EOF

cat >nbody_weak_mpi_${i}.job<<EOF
#!/bin/bash
#SBATCH -J nbody_weak_mpi_${i}
#SBATCH -o nbody_weak_mpi_${i}.o%j
#SBATCH -e nbody_weak_mpi_${i}.e%j
#SBATCH -p normal
#SBATCH -N $i
#SBATCH -n $(($i*16))
#SBATCH -t 01:00:00

APP_HOST="$WORK/nbody_demo/stampede/run_nbody_mpi_weak_host.sh"

export SCALE_PROCS=$(($i *16))
ibrun \$APP_HOST

EOF

cat >nbody_weak_mpi_symm_${i}.job<<EOF
#!/bin/bash
#SBATCH -J nbody_weak_mpi_symm_${i}
#SBATCH -o nbody_weak_mpi_symm_${i}.o%j
#SBATCH -e nbody_weak_mpi_symm_${i}.e%j
#SBATCH -p normal-mic
#SBATCH -N $i
#SBATCH -n $(($i*8))
#SBATCH -t 01:00:00

source $WORK/nbody_demo/stampede/mpi_config.sh
APP_HOST="$WORK/nbody_demo/stampede/run_nbody_mpi_weak_host.sh"
APP_MIC="$WORK/nbody_demo/stampede/run_nbody_mpi_weak_mic.sh"
export SCALE_PROCS=$(($i *16 + $i * 60))
export MIC_PPN=8
export MIC_OMP_NUM_THREADS=30
export OMP_NUM_THREADS=8
ibrun.symm -c \$APP_HOST -m \$APP_MIC

EOF
    
done

