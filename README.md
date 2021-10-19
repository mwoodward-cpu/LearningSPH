# Learning SPH
Learn-able hierarchy of parameterized Smoothed Particle Hydrodynamics (SPH) models trained using mixed mode AD and Sensitivity Analysis (SA). See our paper at (PROVIDE LINK), for more details. 

## Hierarchy of models

We hard code SPH structure into a hierarchy of parameterized models that includes physics based and NN based parameters. Multilayer Perceptrons (MLPs) with hyperbolic tangent activation functions are used as a universal function approximators embedded within an ODE structure describing the Lagrangian flows. It was found through hyper-parameter tuning that 2 hidden layers were sufficient for each model using a NN. 

## Mixed mode AD with Sensitivity analysis
Sensitivity analysis (SA) is a classical technique found in many applications, such as gradient-based optimization, optimal control, parameter identification, model diagnostics; which was also utilized recently to  learn neural network parameters within ODEs (such as in Neural ODE). In the context of this work, we use SA to compute gradients of our hierarchy of parameterized models. We mix SA with Automatic Differentiation (AD);  forward mode and reverse mode AD is applied to derivatives within the SA algorithm, where the method is chosen based on efficiency (depending on the dimension of the input and output space of the function being differentiated).

The code is found in sensitivity_(model).jl, which combines both SA and AD. 

## Loss functions
We construct three different loss functions: trajectory based (Lagrangian), field based Eulerian, and Lagrangian statistics based, all found in the loss_functions.jl file. Since our overall goal involves learning SPH models for turbulence applications, it is the underlying statistical features and large scale field structures we want our models to learn and generalize with.


## Running code
We provide some general guidance for reproduciblity.


### Generate SPH data
in 3d(or 2d)_phys_semi_inf directories, there is a julia file sph_av_3d.jl (or sph_2d_av_turb_ke.jl) for simulating Eulers equations with an Artificial viscosity form and using weakly compressible formulation; see our paper for more details. Parameters of the simulator are commented. Note that some SPH flow data is already provided in the data directories (where parameters are selected as described in the paper). 

### Learning algorithm
The main.jl file consists of our mixed mode learning algorithm. To select which SA method is used, uncomment either sens_method = "forward" or sens_method="adjoint". The loss method is selected simlarly between a combination of the loss functions described above (we find the best results, and for those reasons discussed further in the paper, that loss_method = "kl_lf" performs best; which is the lagrangian statistical based loss using the KL-divergence + the field based loss function using the MSE of difference in velocity fields. The model is selected for by uncommenting "method". Once the model, and methods are choosen for learning, along with the hyper-parameters of the learning, then the ground truth SPH data must be loaded. Running the main.jl file will output all necessary data that can be used in the post_processing directory.

## Paper

for more details see our paper at (PROVIDE LINK).

citation:
(PROVIDE BITEX CITATION)
