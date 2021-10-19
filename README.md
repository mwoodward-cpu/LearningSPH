# Learning SPH
Learn-able hierarchy of parameterized Smoothed Particle Hydrodynamics models trained using mixed mode AD and Sensitivity Analysis (SA). See our paper at (PROVIDE LINK), for more details. 

## Hierarchy of models

We hard code SPH structure into a hierarchy of parameterized models that includes physics based and NN based parameters. Multilayer Perceptrons (MLPs) with hyperbolic tangent activation functions are used as a universal function approximators embedded within an ODE structure describing the Lagrangian flows. It was found through hyper-parameter tuning that 2 hidden layers were sufficient for each model using a NN. 

## Mixed mode AD with Sensitivity analysis
Sensitivity analysis (SA) is a classical technique found in many applications, such as gradient-based optimization, optimal control, parameter identification, model diagnostics; which was also utilized recently to  learn neural network parameters within ODEs (such as in Neural ODE). In the context of this work, we use SA to compute gradients of our hierarchy of parameterized models. We mix SA with Automatic Differentiation (AD);  forward mode and reverse mode AD is applied to derivatives within the SA algorithm, where the method is chosen based on efficiency (depending on the dimension of the input and output space of the function being differentiated).

The code is found in sensitivity_(model).jl, which combines both SA and AD. 

## Loss functions
We construct three different loss functions: trajectory based (Lagrangian), field based Eulerian, and Lagrangian statistics based, all found in the loss_functions.jl file. Since our overall goal involves learning SPH models for turbulence applications, it is the underlying statistical features and large scale field structures we want our models to learn and generalize with. 