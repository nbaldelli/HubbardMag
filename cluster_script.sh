#!/bin/bash
#SBATCH -N1 --exclusive

module load julia

julia -t 100 TJ_cilinder_cluster.jl