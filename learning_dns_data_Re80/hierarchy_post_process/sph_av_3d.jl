"""
Generating 2D WCSPH (AV) taylor green vortex decay flow
"""

using NPZ, Plots, Statistics, LaTeXStrings
using LinearAlgebra

T = 400
t_save = 1   #initial time for saving

IC = "Vrandn"
vmag = 1.0  #initial magnitude of TG velocity
# IC = "TG"
method = "AV_neg_rel"

#params:
c = 0.9157061661168617;
h = 0.2
α = 0.45216843078299573;
β = 0.3346233846532608;
g = 1.0;
θ = 0.00430899795067121;
cdt = 0.4;
dt = cdt * h / c;
p_gt = [c, 0.0];


D = 3;
gsp = 5; #produces grid of 2^gsp x 2^gsp x 2^gsp number of particles
p = Iterators.product((2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi)
p = vec(collect.(p)) #particles
N = length(p)
m = (2. * pi)^D / N;

# sigma = (10. / (7. * pi * h * h)); #2D normalizing factor
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


function Pres(rho)
  return (c^2 * (rho^g - 1.) / g) ;
end

# P(rho) / rho^2
function P_d_rho2(rho)
  return (c^2 * (rho^g - 1.) / g) / (rho^2);
end

function compute_ca(rho, c, g)
  return c * rho^(0.5*(g - 1))
end

function compute_Π(XX, VV, rhoi, rhoj, α, β, h, c)
  ci = compute_ca(rhoi, c, g);
  cj = compute_ca(rhoj, c, g);
  c_bar = (ci + cj)/2;
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    Π = (-α*c_bar*μ + β * μ^2)/((rhoi + rhoj)/2)
  end
  if (XX'*VV >= 0)
    Π = 0.0
  end
  return Π
end

#NEIGHBORHOOD LIST
# The whole (2 pi)^2 torus is divided into n_hash^2 squares with the
# size l_hash. We have l_hash >= h. The particles interact if they
# differ in their "hash" coordinates no more than 2 (i.e., r < 2 h).
n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;

function obtain_sph_av_A(X, V, α, β, h, c)
  μ = 0.0
  Π = 0.0
  A = zeros(N,D); vort = zeros(N,D);
  rho = zeros(N);
  mPdrho2 = zeros(N);
  # tke = 0.5*mean(V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2);

# putting coordinates inside the (2 pi)^2 torus, building hash
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash, k in 1 : n_hash];
  for n in 1 : N
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1,
               floor(Int, X[n, 3] / l_hash) + 1], n);
  end
# computing rho
  XX = zeros(D);
  for n in 1 : N
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1,
              floor(Int, X[n, 3] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
      xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
      while (xb_hash > n_hash)    xb_hash -= n_hash;   end
      for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
        yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
        while (yb_hash > n_hash)    yb_hash -= n_hash;   end
        for za_hash in x_hash[3] - 2 : x_hash[3] + 2
          zb_hash = za_hash;    while (zb_hash < 1)    zb_hash += n_hash;   end
          while (zb_hash > n_hash)    zb_hash -= n_hash;   end
          for n2 in hash[xb_hash, yb_hash, zb_hash]

            close = true; r2 = 0.;
            for i in 1 : D
              XX[i] = X[n, i] - X[n2, i];
              while (XX[i] > pi)   XX[i] -= 2. * pi;   end
              while (XX[i] < -pi)   XX[i] += 2. * pi;   end
              r2 += XX[i] * XX[i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
            if (close)
              tmp = m * W(sqrt(r2), h); rho[n] += tmp;
            end
          end
        end
    end   end
  end
  for n in 1 : N
    mPdrho2[n] = m * P_d_rho2(rho[n]);
  end

  tke = 0.5*mean(rho .* (V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2));
  XX = zeros(D); VV = zeros(D);
  # computing A
  for n in 1 : N
    # A[n, :] = obtain_forcing_A(X[n, :]);
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1,
              floor(Int, X[n, 3] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
      xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
      while (xb_hash > n_hash)    xb_hash -= n_hash;   end
      for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
        yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
        while (yb_hash > n_hash)    yb_hash -= n_hash;   end
        for za_hash in x_hash[3] - 2 : x_hash[3] + 2
          zb_hash = za_hash;    while (zb_hash < 1)    zb_hash += n_hash;   end
          while (zb_hash > n_hash)    zb_hash -= n_hash;   end
          for n2 in hash[xb_hash, yb_hash, zb_hash]

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

              for i in 1 : D
                A[n, i] += tmp * XX[i] #+ θ * (V[n, i] - mean(V[:, i]));
                vort[n, i] += -m/rho[n] * cross(VV, XX)[i] * H(sqrt(r2), h);
              end
            end
          end
        end
    end   end
  end
  for i in 1 : D
    A[:, i] += (θ/(2*tke)) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho, vort
end



#---------------Initial conditions (and boundary condistion)
# include("./data_loader.jl")
#
# pos_path = "./wc_dns_4096_unif_longer_time/pos_traj_4k.npy"
# vel_path = "./wc_dns_4096_unif_longer_time/vel_traj_4k.npy"
# rho_path = "./wc_dns_4096_unif_longer_time/rho_traj_4k.npy"
#
# traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)

function obtain_vrand_ic()
  X = zeros(N, D); V = zeros(N, D);
  for n in 1 : N
    X[n, 1] = p[n][1] + 0.0005 * (rand() - 0.5)
    X[n, 2] = p[n][2] + 0.0005 * (rand() - 0.5)
    X[n, 3] = p[n][3] + 0.0005 * (rand() - 0.5)
  end
  X = mod.(X, 2*pi);
  # for n in 1 : N
  #   #Taylor green intitial condition
  #   V[n, 1] = vmag * sin(X[n, 1]) * cos(X[n, 2]) * cos(X[n, 3])
  #   V[n, 2] = -vmag * cos(X[n, 1]) * sin(X[n, 2]) * cos(X[n, 3]);
  #   for i in 1 : D
  #     while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
  #     while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
  #   end
  # end

  V = vmag * randn(N, D)
  return X, V
end


#-------------- Integration

function vel_verlet(α, β, h, c, T)
  """
  velocity verlet without external forcing
    returns trajectories, velocitys and densities of particles
  """

  # X, V = traj_gt[1,:,:], vels_gt[1,:,:]
  X, V = obtain_vrand_ic()

  traj, vels = zeros(T+1,N,D), zeros(T+1,N,D); rhos = zeros(T+1, N);
  Vort = zeros(T+1,N,D);
	traj[1, :, :] = X; vels[1, :, :] = V;
	A, rho = obtain_sph_av_A(X, V, α, β, h, c);
	rhos[1,:] = rho;

	for k in 1 : T
	  A, rho, vort = obtain_sph_av_A(X, V, α, β, h, c)

	  #Verlet
	  for n in 1 : N
  		for i in 1 : D
			V[n, i] += 0.5 * dt * A[n, i];
			X[n, i] += dt * V[n, i];
			while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
			while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
  		end
	  end

	  A, rho, vort = obtain_sph_av_A(X, V, α, β, h, c)

	  for n in 1 : N   for i in 1 : D
		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
	  end   end

	  vels[k + 1, :, :] = V;
	  traj[k + 1, :, :] = X;
	  rhos[k + 1, :] = rho;
    Vort[k+1, :, :] = vort;
	  println("time step:  ", k)
	end
	return traj, vels, rhos, Vort
end


traj, vels, rhos, Vort = vel_verlet(α, β, h, c, T)
Vort_field = 0.5 * (Vort[T,:,1].^2 .+ Vort[T,:,2].^2 .+ Vort[T,:,3].^2)

# include("gen_vorticity_outs.jl")
#
# r = 0.2; v = 1.3*r
# iso_surface_vorticity(Vort_field, r, v)


function simulate(pos, sim_time=15)
    gr(size=(1000,800))
    sim_path = "./learned_sims/traj_N$(N)_T$(T)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ).mp4"
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2); m_s = 1.75
    anim = @animate for i ∈ 1:(T+1)
         println("sim time = ", i)
         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
         title = "Learned WCSPH_$(method): N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms = m_s)
         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red", ms = m_s)
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end

simulate(traj)

function save_data_files(vort, traj, vels, rhos, t_save)
    println(" ****************** Saving data files ***********************")
    vor_path = "./learned_data/vort_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
    pos_path = "./learned_data/traj_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
    vel_path = "./learned_data/vels_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
    rho_path = "./learned_data/rhos_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"

	npzwrite(vor_path, vort[t_save:end,:,:])
	npzwrite(pos_path, traj[t_save:end,:,:])
	npzwrite(vel_path, vels[t_save:end,:,:])
	npzwrite(rho_path, rhos[t_save:end,:])
end

save_data_files(Vort, traj, vels, rhos, 1)
