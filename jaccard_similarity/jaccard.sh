#!/bin/bash
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -t 3:00:00
#SBATCH -p single
#SBATCH -A loni_tsr_9

module purge
module load python/3.7.6
module load hdf5/1.10.6/intel-19.0.5-mvapich-2.3.3
export PATH=/project/wxx6941/packages/par_version.qb3:$PATH

# fill in absolute path here for input, *must* contain the csv file
input_sample_folder="./jaccard_similarity"
# fill in absolute path here for output
#output_folder="./theta29_dist18"
output_folder="./theta29_dist18"

# remove possible tailing slashes
output_folder=${output_folder%/}
input_sample_folder=${input_sample_folder%/}

lf_csv2_dtype_h5.py -f "$input_sample_folder" -o "$output_folder"

input_sample_name=${input_sample_folder##*/}
echo "input_sample_name=$input_sample_name"

input_h5="${output_folder}/${input_sample_name}.h5"
echo "input_h5=$input_h5"

export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=6
SECONDS=0
#mpirun -np $NPROCS -f $PBS_NODEFILE -ppn 2 intel.mpi.omp.out -f $input_h5
srun --overlap -n 8 intel.mpi.omp.out -f $input_h5
echo "omp run threads=$OMP_NUM_THREADS, $SECONDS sec"

SECONDS=0
rebuild_mat.py -f $input_h5 -csv #-validate
echo "rebuild mat took $SECONDS sec"

