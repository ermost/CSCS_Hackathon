#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1 
#SBATCH --ntasks=5 
#SBATCH --cpus-per-task=2 
#SBATCH --gres=gpu:1 
#SBATCH --time=4:00:00

module purge > /dev/null 2>&1

module load compilers/gcc-7.5.0
module load cuda

module load gcc/7.5.0/openmpi/4.0.5-gpu

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=5

mpirun -np 1 --map-by ppr:1:node:PE=5 --mca btl ^openib --report-bindings nsys profile ./CNS3d.gnu.TPROF.MPI.CUDA.ex inputs-rt 2>&1 | tee out.${SLURM_JOBID}
