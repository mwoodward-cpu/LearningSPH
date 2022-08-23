#!/bin/bash
#SBATCH --job-name=mt08_node
#SBATCH --nodes=1 --ntasks=1
#SBATCH --constraint=hi_mem 
#SBATCH --time=10:00:00
#SBATCH --output=mt08_node%A.out
#SBATCH --account=chertkov                                                                                     
#SBATCH --partition=standard
#SBATCH --array=1-5
echo "$SLURM_ARRAY_TASK_ID"

# --------------------------------------------------------------                                                        
### PART 2: Executes bash commands to run your job                                                                      
# --------------------------------------------------------------                                                        
### Load required modules/libraries if needed                                                                           
module load julia/1.6.1
### change to your script’s directory                                                                                   

### Run your work                                                                                                       
julia ./hpc_code/running_trained_models_long_t_generalization_t50_lf_${SLURM_ARRAY_TASK_ID}.jl 
