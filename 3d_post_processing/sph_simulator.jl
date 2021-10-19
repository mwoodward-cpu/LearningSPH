"""
Generating 2D WCSPH (AV) ground truth "tubulence" data
  External forcing due to negative relaxtion term

  dvi/dt = (Pi/rhoi^2 + Pj/rhoj^2 + Pi)∇Wij + θ(Vi - ̄Vi)
"""



#------SPH functions
# sigma = (10. / (7. * pi * h * h)); #for 2d
sigma = 1/(pi*h^3)  #3D normalizing factor
function W(r, h)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

# H(r) = (d W / d r) / r
function H(r, h)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * (-3. + 9. * q / 4.) / h^2);
end


function Pres(rho, c)
  return (c^2 * (rho^g - 1.) / g) ;
end

# P(rho) / rho^2
function P_d_rho2(rho, c)
  return (c^2 * (rho^g - 1.) / g) / rho^2;
end


#artificial viscosity term Πᵢⱼ
function compute_Π(XX, VV, rhoi, rhoj, α, β, h, c)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    Π = (-α*c*μ + β * μ^2)/((rhoi + rhoj)/2)
  end
  if (XX'*VV >= 0)
    Π = 0.0
  end
  return Π
end



#------------Computing sum with O(N) Neighborhood list

# The whole (2 pi)^2 torus is divided into n_hash^2 squares with the
# size l_hash. We have l_hash >= h. The particles interact if they
# differ in their "hash" coordinates no more than 2 (i.e., r < 2 h).
n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;

function obtain_sph_av_A(X, V, α, β, h, c, g, θ)
  μ = 0.0
  Π = 0.0
  A = zeros(N,D);
  rho = zeros(N);
  mPdrho2 = zeros(N);

  tke = 0.5*mean(V[: ,1].^2 .+ V[: ,2].^2)
# putting coordinates inside the (2 pi)^2 torus, building hash
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash];
  for n in 1 : N
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1], n);
  end
# computing rho
  for n in 1 : N
    rho[n] = 0.;
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
      xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
      while (xb_hash > n_hash)    xb_hash -= n_hash;   end
      for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
        yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
        while (yb_hash > n_hash)    yb_hash -= n_hash;   end
        for n2 in hash[xb_hash, yb_hash]
          close = true; r2 = 0.;
          for i in 1 : D
            XX = X[n, i] - X[n2, i];
            while (XX > pi)   XX -= 2. * pi;   end
            while (XX < -pi)   XX += 2. * pi;   end
            r2 += XX * XX;
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)
            tmp = m * W(sqrt(r2), h); rho[n] += tmp;
          end
        end
    end   end
  end
  for n in 1 : N
    mPdrho2[n] = m * P_d_rho2(rho[n], c);
  end

  XX = zeros(D); VV = zeros(D);
  # computing A
  for n in 1 : N
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
      xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
      while (xb_hash > n_hash)    xb_hash -= n_hash;   end
      for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
        yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
        while (yb_hash > n_hash)    yb_hash -= n_hash;   end
        for n2 in hash[xb_hash, yb_hash]
          close = true;   r2 = 0.;
          for i in 1 : D
            XX[i] = X[n, i] - X[n2, i];
            VV[i] = V[n, i] - V[n2, i];
            while (XX[i] > pi)   XX[i] -= 2. * pi;   end
            while (XX[i] < -pi)   XX[i] += 2. * pi;   end
            r2 += XX[i] * XX[i];
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)
            Π = compute_Π(XX, VV, rho[n], rho[n2], α, β, h, c)
            tmp = -(mPdrho2[n] + mPdrho2[n2] + m*Π) * H(sqrt(r2), h);
            for i in 1 : D   A[n, i] += tmp * XX[i];   end
          end
        end
    end   end
  end
  for i in 1 : D
      A[:, i] += (θ/tke) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho
end



#-------------- Integration (Velocity Verlet)

function vel_verlet(traj_gt, vels_gt, α, β, h, c, g, θ, T)
  """
  velocity verlet (RHS depends on V) with external forcing
    returns trajectories, velocitys and densities of particles
  """
  X = zeros(N, D); V = zeros(N, D);
  @inbounds for n in 1 : N,  i in 1 : D
	X[n,i] = traj_gt[1, n, i]
	V[n,i] = vels_gt[1, n, i]
  end

  A₀, ρ₀ = obtain_sph_av_A(X, V, α, β, h, c, g, θ)

  traj, vels, dens = zeros(T+1,N,D), zeros(T+1,N,D), zeros(T+1,N);
  c_s = zeros(T);

  traj[1, :, :] = X; vels[1, :, :] = V
  dens[1,:] = ρ₀

	t = 0.;
	for k in 1 : T
	  A, rho = obtain_sph_av_A(X, V, α, β, h, c, g, θ)

	  #Verlet
	  for n in 1 : N
  		for i in 1 : D
			V[n, i] += 0.5 * dt * A[n, i];
			X[n, i] += dt * V[n, i];
			while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
			while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
  		end
	  end

	  A, rho = obtain_sph_av_A(X, V, α, β, h, c, g, θ)

	  for n in 1 : N   for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
	  end   end

	  vels[k + 1, :, :] = V
	  traj[k + 1, :, :] = X
      dens[k + 1, :] = rho
      c_s[k] = maximum(V)/sqrt(0.01)
	  println("time step:", k)
	  t += dt;
	end
	return traj, vels, dens
end



function vel_verlet_NN(traj_gt, vels_gt, p, α, β, h, c, g, θ, T)
  """
  velocity verlet (RHS depends on V) with external forcing
    returns trajectories, velocitys and densities of particles
  """
  X = zeros(N, D); V = zeros(N, D);
  traj, vels, rhos = zeros(T+1,N,D), zeros(T+1,N,D), zeros(T+1,N);
  @inbounds for n in 1 : N,  i in 1 : D
	X[n,i] = traj_gt[1, n, i]
	V[n,i] = vels_gt[1, n, i]
  end

  traj[1, :, :] = X; vels[1, :, :] = V;
  A, rho = obtain_sph_AV_A(X, V, p, c, h, α, β, θ)
  rhos[1, :] = rho;

  t = 0.;
  for k in 1 : T
	A, rho = obtain_sph_AV_A(X, V, p, c, h, α, β, θ)

	#Verlet
	for n in 1 : N
	  for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i];
		  X[n, i] += dt * V[n, i];
		  while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
		  while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
	  end
	end

	A, rho = obtain_sph_AV_A(X, V, p, c, h, α, β, θ)

	for n in 1 : N
		for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
		end
	end

	traj[k + 1, :, :] = X
	vels[k + 1, :, :] = V;
	rhos[k + 1, :, :] = rho;
	println("time step:", k)
	t += dt;
  end
  return traj, vels, rhos
end
