#!/bin/bash
#SBATCH --job-name=mt08_re80
#SBATCH --nodes=1 --ntasks=2
#SBATCH --constraint=hi_mem 
#SBATCH --time=240:00:00
#SBATCH --output=mt08_re80%A.out
#SBATCH --account=chertkov                                                                                     
#SBATCH --partition=standard
#SBATCH --array=6
echo "$SLURM_ARRAY_TASK_ID"

# --------------------------------------------------------------                                                        
### PART 2: Executes bash commands to run your job                                                                      
# --------------------------------------------------------------                                                        
### Load required modules/libraries if needed                                                                           
module load julia/1.6.1
### change to your script’s directory                                                                                   

### Run your work                                                                                                       
julia ./main4_${SLURM_ARRAY_TASK_ID}_cont2.jl lf forward 0 unif_tracers 20 600 1

### Loss, sens_method, switch_loss, ic, T, itrs, t_coarse
