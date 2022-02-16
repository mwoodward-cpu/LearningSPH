"""
Generating 2D WCSPH (AV) ground truth "tubulence" data
  External forcing due to negative relaxtion term

  dvi/dt = (Pi/rhoi^2 + Pj/rhoj^2 + Pi)∇Wij + θ(Vi - ̄Vi)
"""

using Plots, StatsPlots
using NPZ, Statistics
using LaTeXStrings


T = 800
t_save = 1      #(time step expected to be stationary beyond)
mag = 1.0         #magnitude of external forcing
t_start = 150     #initial condition from stationary data


IC = "Vrandn"
# IC = "s_hit"       #uniform distribution of particles
#IC = "shear"
method = "AV_neg_rel_ke"     #Artificial viscosity

#AV params:
α = 1e0           #bulk and shear viscosity parameter
β = 2.0 * α           #Neuman Richtmyer viscosity parameter
θ = 4e-1          #negative relaxation term (extern forcing)


c = 10.0          #artificial speed of sound
g = 7.0           #γ: stiffing EoS (see Batchlor)
h = 0.2         #smoothing length
cdt = 0.4;        #factor for CFL
dt = cdt * h/c;   #dt considering CFL

power = 10
D = 2; N = round(Int, 2^power); # dimension and number of particles
m = (2. * pi)^D / N;


#------SPH functions
sigma = (10. / (7. * pi * h * h)); #for 2d
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
function obtain_sph_av_A(X, V, α, β, h, c)
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
    mPdrho2[n] = m * P_d_rho2(rho[n]);
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



#---------------Initial conditions (and boundary condition)


function obtain_unif_ic()
  X₀ = zeros(N, D); V₀ = zeros(N, D);
  for i in 1 : round(Int, 2^(power/2))   for j in 1 : round(Int, 2^(power/2))
    n = round(Int, 2^(power/2)) * (i - 1) + j;
    X₀[n, 1] = 2. * pi * ((i - 1.) / sqrt(N) + 0.001 * (rand() - 0.5));
    X₀[n, 2] = 2. * pi * ((j - 1.) / sqrt(N) + 0.001 * (rand() - 0.5));
  end   end
  X₀ = mod.(X₀, 2*pi);  #periodic BCs
  return X₀, V₀
end

function obtain_s_hit_ic()
	X₀ = npzread("../data/pos_N1024_T12000_ts11800_g7_h0.25_cubic_alpha-0.05_beta0.1_10mode_tfreq2___cdt0.4_mag1.0_vmag0.0_C30_AV_fixedc.npy")
	V₀ = npzread("../data/vel_N1024_T12000_ts11800_g7_h0.25_cubic_alpha-0.05_beta0.1_10mode_tfreq2___cdt0.4_mag1.0_vmag0.0_C30_AV_fixedc.npy")
	return X₀[t_start,:,:], V₀[t_start,:,:]
end

function obtain_shear_ic()
  X₀ = zeros(N, D); V₀ = zeros(N, D);
  for i in 1 : round(Int, 2^(power/2))   for j in 1 : round(Int, 2^(power/2))
    n = round(Int, 2^(power/2)) * (i - 1) + j;
    X₀[n, 1] = 2. * pi * ((i - 1.) / sqrt(N) + 0.001 * (rand() - 0.5));
    X₀[n, 2] = 2. * pi * ((j - 1.) / sqrt(N) + 0.001 * (rand() - 0.5));
  end   end
  X₀ = mod.(X₀, 2*pi);  #periodic BCs
  for n in 1 : N
      if n < (N/2)
      	 V₀[n, 2] = 2.0;
      else
	 V₀[n, 2] = -2.0;
      end
   end
  return X₀, V₀
end



#-------------- Integration

function vel_verlet(α, β, h, c, T)
  """
  velocity verlet (RHS depends on V) with external forcing
    returns trajectories, velocitys and densities of particles
  """

  if IC=="Vrandn"
	  X, V = obtain_unif_ic()   #initial conditions
	  V = randn(N, D);
  end
  if IC=="s_hit"
  	X, V = obtain_s_hit_ic()   #initial conditions
  end
  if IC=="shear"
     X, V = obtain_shear_ic()
  end
  A₀, ρ₀ = obtain_sph_av_A(X, V, α, β, h, c)

  traj, vels, dens = zeros(T+1,N,D), zeros(T+1,N,D), zeros(T+1,N);
  KE_fluc = zeros(T);
  c_s = zeros(T);

  traj[1, :, :] = X; vels[1, :, :] = V
  dens[1,:] = ρ₀

	t = 0.;
	for k in 1 : T
	  A, rho = obtain_sph_av_A(X, V, α, β, h, c)

	  #Verlet
	  for i in 1 : D
  		for n in 1 : N
			V[n, i] += 0.5 * dt * A[n, i];
			X[n, i] += dt * V[n, i];
			while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
			while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;   end
  		end
	  end

	  A, rho = obtain_sph_av_A(X, V, α, β, h, c)

	  for i in 1 : D   for n in 1 : N
		  V[n, i] += 0.5 * dt * A[n, i] #+ stochastic_term();
	  end   end

	  vels[k + 1, :, :] = V
	  traj[k + 1, :, :] = X
	  KE_fluc[k] = mean(0.5 * ((V[:,1] .- mean(V[:,1])).^2 + (V[:,2] .- mean(V[:,2])).^2))
      dens[k + 1, :] = rho
      c_s[k] = maximum(V)/sqrt(0.01)
	  println("time step:", k)
	  t += dt;
	end

	return traj, vels, dens, KE_fluc
end


traj, vels, rhos, KE_fluc = vel_verlet(α, β, h, c, T)



function obtain_tke(vels)
  tke = zeros(T);
  for k in 1 : T
      tke[k] = 0.5*mean(vels[k,:,1].^2 .+ vels[k,:,2].^2)
  end
  return tke
end

tke = obtain_tke(vels)

function compute_Re(vels)
  Re = zeros(T+1)
  V = 0.0
  L = 2*pi
  ν = 1/10 * α * c * h
  for t in 1 : (T+1)
    V = maximum(vels[t, :, :])
    Re[t] = L*V/ν
  end
  return Re
end

Re = compute_Re(vels)

function compute_turb_ke(vels)
	turb_ke = zeros(T);
	for t in 1 : T
		avg_KE = 0.5*mean(vels[t, :, 1].^2 .+ vels[t, :, 2].^2)
		dec_ke = 0.5 * (mean(vels[t, :, 1])^2 + mean(vels[t, :, 2])^2)
		turb_ke[t] = avg_KE - dec_ke
	end
	return turb_ke
end

turb_ke = compute_turb_ke(vels)

#-----------Generating outputs

ENV["GKSwstype"]="100"

function make_dir(path)
	if isdir(path) == true
	       println("directory already exists")
	   else mkdir(path)
	end
end

make_dir("figures"); make_dir("sims"); make_dir("data")


#-----------Outputs
pos_path = "./data/traj_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).npy"
vel_path = "./data/vels_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).npy"
rho_path = "./data/rhos_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).npy"

sim_path = "./sims/traj_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).mp4"
file_out_tke = "./figures/tke_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
file_out_re = "./figures/re_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
file_out_turb_ke = "./figures/turb_ke_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"


function plot_KE_fluc()
    gr(size=(500,400))
    println("*************** generating plots ******************")

    plt3 = plot(tke, label=L"Avg(KE)", color="blue", linewidth = 2.25)
    title!("Average kinetic energy, θ = $(θ)")
    xlabel!(L"t")
    ylabel!(L"Avg KE")
    display(plt3)

	plt = plot(turb_ke, label=L"k", color="blue", linewidth = 2.25)
    title!("Turbulent kinetic energy, θ = $(θ)")
    xlabel!(L"t")
    ylabel!(L"turbulent KE")
    display(plt)
    savefig(plt, file_out_turb_ke)
end


function plotting_Re()
  plt = plot(Re, label=L"Re", color="blue", linewidth = 2.25)
  title!("Reynolds number, θ = $(θ)")
  xlabel!(L"t")
  ylabel!(L"Re")
  display(plt)
  savefig(plt, file_out_re)
end


function gen_data_files(traj, vels, rhos)
    println(" ****************** Saving data files ***********************")
    # npzwrite(pos_path, pos[t_save:end,:,:])
    # npzwrite(vel_path, vel[t_save:end,:,:])
	npzwrite(pos_path, traj[t_save:end,:,:])
	npzwrite(vel_path, vels[t_save:end,:,:])
	npzwrite(rho_path, rhos[t_save:end,:])
end



function simulate(pos, sim_time=10)
    gr(size=(1000,800))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1:2:(T+1)
         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2],
         title = "WCSPH_$(method): N=$(N), h=$(h), c=$(c)", xlims = [0, 2*pi], ylims = [0,2*pi], legend = false)
         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], color = "red")
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end

function simulate_same_color(pos, sim_time=8)
    gr(size=(1000,800))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1:2:(T+1)
         Plots.scatter(pos[i, :, 1], pos[i, :, 2],
         title = "WCSPH_$(method): N=$(N), h=$(h), c=$(c)", xlims = [0, 2*pi],
         ylims = [0,2*pi], legend = false, color="blue")
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end



#UAHPC seems to need this formatting
ENV["GKSwstype"]="100"

gen_data_files(traj, vels, rhos)
plot_KE_fluc()
plotting_Re()
simulate(traj, 40)
# simulate_same_color(traj, 8)
