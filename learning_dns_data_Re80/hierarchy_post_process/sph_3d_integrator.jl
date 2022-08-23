"""
Generating 2D WCSPH (AV) ground truth "tubulence" data
  External forcing due to negative relaxtion term

  dvi/dt = (Pi/rhoi^2 + Pj/rhoj^2 + Pi)∇Wij + θ(Vi - ̄Vi)
"""





function vel_verlet(traj_gt, vels_gt, p, T)
  """
  velocity verlet (RHS depends on V) with external forcing
    returns trajectories, velocitys and densities of particles
  """
  X = zeros(N, D); V = zeros(N, D);
  traj, vels, rhos = zeros(T+1,N,D), zeros(T+1,N,D), zeros(T+1,N);
  accl = zeros(T+1, N, D);
  @inbounds for n in 1 : N,  i in 1 : D
	X[n,i] = traj_gt[1, n, i]
	V[n,i] = vels_gt[1, n, i]
  end

  traj[1, :, :] = X; vels[1, :, :] = V;
  A, rho = obtain_sph_AV_A(X, V, p)
  rhos[1, :] = rho;
  accl[1,:,:] = A;
  for k in 1 : T
	A, rho = obtain_sph_AV_A(X, V, p)

	#Verlet
	for n in 1 : N
	  for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i];
		  X[n, i] += dt * V[n, i];
		  while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
		  while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
	  end
	end

	A, rho = obtain_sph_AV_A(X, V, p)

	for n in 1 : N
		for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
		end
	end

	accl[k+1,:,:] = A;
	traj[k + 1, :, :] = X
	vels[k + 1, :, :] = V;
	rhos[k + 1, :, :] = rho;
	# Vor[k + 1, :, :] = vort;
	println("time step:", k)
  end
  return accl, traj, vels, rhos
end








# function vel_verlet(traj_gt, vels_gt, p, T)
#   """
#   velocity verlet (RHS depends on V) with external forcing
#     returns trajectories, velocitys and densities of particles
#   """
#   X = zeros(N, D); V = zeros(N, D);
#   traj, vels, rhos = zeros(T+1,N,D), zeros(T+1,N,D), zeros(T+1,N);
#   @inbounds for n in 1 : N,  i in 1 : D
# 	X[n,i] = traj_gt[1, n, i]
# 	V[n,i] = vels_gt[1, n, i]
#   end
#
#   traj[1, :, :] = X; vels[1, :, :] = V;
#   A, rho = obtain_sph_AV_A(X, V, p)
#   rhos[1, :] = rho;
#
#   for k in 1 : T
# 	A, rho = obtain_sph_AV_A(X, V, p)
#
# 	#Verlet
# 	for n in 1 : N
# 	  for i in 1 : D
# 		  V[n, i] += 0.5 * dt * A[n, i];
# 		  X[n, i] += dt * V[n, i];
# 		  while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
# 		  while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
# 	  end
# 	end
#
# 	A, rho = obtain_sph_AV_A(X, V, p)
#
# 	for n in 1 : N
# 		for i in 1 : D
# 		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
# 		end
# 	end
#
# 	traj[k + 1, :, :] = X
# 	vels[k + 1, :, :] = V;
# 	rhos[k + 1, :, :] = rho;
# 	println("time step:", k)
#   end
#   return traj, vels, rhos
# end
