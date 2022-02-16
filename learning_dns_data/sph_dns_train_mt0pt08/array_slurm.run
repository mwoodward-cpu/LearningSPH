#!/bin/bash
#SBATCH --job-name=mt08_node
#SBATCH --nodes=1 --ntasks=2
#SBATCH --constraint=hi_mem 
#SBATCH --time=240:00:00
#SBATCH --output=mt08_node%A.out
#SBATCH --account=chertkov                                                                                     
#SBATCH --partition=standard
#SBATCH --array=5-6
echo "$SLURM_ARRAY_TASK_ID"

# --------------------------------------------------------------                                                        
### PART 2: Executes bash commands to run your job                                                                      
# --------------------------------------------------------------                                                        
### Load required modules/libraries if needed                                                                           
module load julia/1.6.1
### change to your script’s directory                                                                                   

### Run your work                                                                                                       
julia ./main4_${SLURM_ARRAY_TASK_ID}.jl lf forward 0 unif_tracers 30 600
