
#---------integrators
function RK4_step(S_i, ∂F_x, ∂F_p, RHS_S)
	#TODO use ode package to implement forward steps of sensitiviites
	H_θ_update = zeros(N, 2*D, n_params);
	for i ∈ 1 : N
		k1 = dt * RHS_S(S_i[i,:,:], ∂F_x, ∂F_p, i);
		k2 = dt * RHS_S(S_i[i,:,:] + 0.5 * k1, ∂F_x, ∂F_p, i);
		k3 = dt * RHS_S(S_i[i,:,:] + 0.5 * k2, ∂F_x, ∂F_p, i);
		k4 = dt * RHS_S(S_i[i,:,:] + k3, ∂F_x, ∂F_p, i);
		H_θ_update[i, :, :] = (k1 + 2. * k2 + 2. * k3 + k4)/6
	end
	return H_θ_update
end

function rhsS_i(S_i, ∂F_x, ∂F_p, i)
  """
  RHS of dS_i/dt = ∂F_x_i * S_i + ∂F_p_i
  returns i'th component matrix of size (D^2, n_params)
  """

  ∂F_p[i, 1, :] = S_i[3, :]
  ∂F_p[i, 2, :] = S_i[4, :]
  ∂F_x[i, 1, :] = [0.0, 0.0, 1.0, 0.0]
  ∂F_x[i, 2, :] = [0.0, 0.0, 0.0, 1.0]
  return ∂F_x[i, :, :] * S_i + ∂F_p[i, :, :]
end


function simultaneous_integration(p, T)
	traj = zeros(T+1,N,D); vels = zeros(T+1,N,D);
	rhos = zeros(T+1,N);
	S = zeros(N, 2*D, n_params); #initial conditions
	ST = zeros(T+1, N, 2*D, n_params);
	X = zeros(N, D); V = zeros(N, D);
	for i in 1 : D, n in 1 : N
		X[n,i] = traj_gt[1, n, i]
		V[n,i] = vels_gt[1, n, i]
	end
	traj[1, :, :] = X; vels[1, :, :] = V;
	A, rho = obtain_sph_AV_A(X, V, p)
	rhos[1, :] = rho;

	for k in 1 : T
	  A, rho, ∂F_x, ∂F_p = obtain_sph_diff_A_model_deriv(X, V, p)

	  #Verlet
	  for i in 1 : D, n in 1 : N
			V[n, i] += 0.5 * dt * A[n, i];
			X[n, i] += dt * V[n, i];
			while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
			while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
	  end

	  A, rho = obtain_sph_AV_A(X, V, p)

	  for i in 1 : D, n in 1 : N
		  	V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
	  end

	  S = RK4_step(S, ∂F_x, ∂F_p, rhsS_i) #simultaenously solve CFSA
	  traj[k + 1, :, :] = X
	  vels[k + 1, :, :] = V;
	  rhos[k + 1, :, :] = rho;
	  ST[k + 1, :, :, :] = S
	  # println("time step:", k)
	end
	return traj, vels, rhos, ST
end




function dual_adjoint_integration(p, T)
	traj = zeros(T+1,N,D); vels = zeros(T+1,N,D);
	rhos = zeros(T+1,N);
	H = zeros(N, 2*D, n_params); #initial conditions
	ST = zeros(T+1, N, 2*D, n_params);
	X = zeros(N, D); V = zeros(N, D);
	for i in 1 : D, n in 1 : N
		X[n,i] = traj_gt[1, n, i]
		V[n,i] = vels_gt[1, n, i]
	end

	traj[1, :, :] = X; vels[1, :, :] = V;

	∂F_xT = zeros(T+1, N, 2*D, 2*D);
	∂F_pT = zeros(T+1, N, 2*D, n_params);
	λT = zeros(T+1, N, 2*D);
	∂L_x = zeros(T+1, N, 2*D);

	#------Verlet integration
	for k in 1 : T
	  A, rho, ∂F_x, ∂F_p = obtain_sph_diff_A_model_deriv(X, V, p)

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

	  for n in 1 : N   for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
	  end   end

	  vels[k + 1, :, :] = V
	  traj[k + 1, :, :] = X
	  rhos[k + 1, :, :] = rho;
	  ∂F_xT[k + 1, :, :, :] = ∂F_x
	  ∂F_pT[k + 1, :, :, :] = ∂F_p

	  diff_step, Vel_inc_pred = obtain_pred_dists_step(X, V, traj[1,:,:], vels[1,:,:])
	  if loss_method=="kl"
	  	∂L_x[k+1, :, :] = ∂kl_∂z(Vel_inc_gt, Vel_inc_pred, k)
	  end
	  if loss_method=="l2"
		for i in 1 : N
			∂L_x[k+1, i, :] = ∂xmseᵢ(X, traj_gt[k+1,:,:], V, vels_gt[k+1,:,:], i)
		end
	  end
	end

	#-----adjoint variable integration (in reverse time) (for now just euler)
	for k2 in (T+1): -1 : 2
		for n in 1 : N
			λT[(k2-1), n, :] = (∂L_x[k2, n, :]' - λT[k2, n, :]' * ∂F_xT[k2, n, :, :])*(dt)
		end
	end
	return traj, vels, rhos, λT, ∂F_pT #, ST_h #pos, vel #Final position so far later use pos, vel
end

function compute_adjoint_∇L(λT, ∂F_pT)
	∇L = zeros(n_params)
	for i in 1 : N
		for τ in 1 : (T+1)
			for k in 1 : n_params
				∇L[k] += dt * λT[τ, i, :]' * ∂F_pT[τ, i, :, k]
			end
		end
	end
	return ∇L
end



function rotational_metric(X, V, p, model_A)
	Q,R = qr(randn(D,D)); Q = Q*Diagonal(sign.(diag(R))); #random orthogonal matrix
	R_90 = [0.0 -1.0; 1.0 0.0];

	Fqy, rh_ = model_A((Q*X')', (Q*V')', p)
	Fry, rh_ = model_A((R_90*X')', (R_90*V')', p)
	F, rh_ = model_A(X, V, p)
	QF = (Q*F')'
	RF = (R_90 * F')'
	return mse(Fqy, QF), mse(Fry, RF)
end

shiftx = 2*pi*rand(); shiftv = 2*pi*rand();
function translational_metric(X, V, p, model_A)
    Fy_s, rh_ = model_A(X .- shiftx, V .- shiftv, p)
	Fx_s, rh_ = model_A(X .- shiftx, V, p)
    F, rh_ = model_A(X, V, p)
    return mse(Fy_s, F), mse(Fx_s, F)
end


#switch to kl + lf loss at some iteration:
function kl_lf_switch(k, switch, l_method, itr, L)
	loss_method = l_method
	if switch == 1
		if k > itr
			loss_method = "kl_lf"
		end
		if loss_method == "kl"
			if L < 2e-3
				loss_method = "kl_lf"
			end
		end
	end
	return loss_method
end
