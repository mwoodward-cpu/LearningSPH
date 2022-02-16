using NPZ, Plots


function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end

make_dir("learned_fields");


#----------Load data
t_start = 1;
θ = 0.0002;
h = 0.335;
t_coarse = 2
dt = t_coarse*0.02;


include("./data_loader.jl")
pos_path = "./wc_dns_unif_4096_gen_data_mt0.08/pos_traj_4k.npy"
vel_path = "./wc_dns_unif_4096_gen_data_mt0.08/vel_traj_4k.npy"
rho_path = "./wc_dns_unif_4096_gen_data_mt0.08/rho_traj_4k.npy"

traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;


#----------GRID

m = (2. * pi)^D / N;
nz = ny = nx = round(Int,(N)^(1/3))
L = Lx = Ly = 2*pi


function obtain_unif_mesh_from_tracers(traj)
	X_grid = traj[1,:,:]
	return X_grid
end
X_grid = obtain_unif_mesh_from_tracers(traj_gt)
# X_grid = reshape(X_grid, (nx, ny, nz, D))

#------SPH interpolation


#smoothing kernel
function W(r, h)
  sigma = 1/(pi*h^3)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end


n_hash = floor(Int, 2*pi / h);   l_hash = 2*pi / n_hash;
function obtain_interpolated_velocity(X_grid, X, V, rho, N_f)
  Vf = zeros(N,D);
  ∂Vf_∂x = zeros(N, D); ∂Vf_∂y = zeros(N, D); ∂Vf_∂z = zeros(N, D);
  ∂Vf_∂u = zeros(N, D); ∂Vf_∂v = zeros(N, D); ∂Vf_∂w = zeros(N, D);
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash, k in 1 : n_hash];
  for n in 1 : N_f
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1,
               floor(Int, X[n, 3] / l_hash) + 1], n);
  end

  XX = zeros(D);
  for n in 1 : N
    x_hash = [floor(Int, X_grid[n, 1] / l_hash) + 1,
  			floor(Int, X_grid[n, 2] / l_hash) + 1,
  			floor(Int, X_grid[n, 3] / l_hash) + 1];
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
            XX[i] = X_grid[n, i] - X[n2, i];
            while (XX[i] > pi)   XX[i] -= 2. * pi;   end
            while (XX[i] < -pi)   XX[i] += 2. * pi;   end
            r2 += XX[i] * XX[i];
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)
			for i in 1 : D
	            Vf[n, i] += m * W(sqrt(r2), h) * V[n2, i] / rho[n2];
			end
          end
        end
    end   end   end
  end
  Vf = reshape(Vf, (nx, ny, nz, D))
  return Vf
end



function obtain_interpolated_velocity_over_t(X_grid, traj, vels, rhos, t_s)
	Vf_t = zeros(t_s+1,nx,ny,nz,D);
	for t in 1 : t_s
		println("Time step =  ", t)
		Vf_t[t,:,:,:,:] = obtain_interpolated_velocity(X_grid, traj[t,:,:], vels[t,:,:], rhos[t,:], N)
	end
	return Vf_t
end

function load_sim_coarse_data(method, l_method, Tt, itr, lr)
	data_dir_phy = "./learned_data/"
	acc_path = "$(data_dir_phy)/accl_Tp250_Tt$(Tt)_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	tra_path = "$(data_dir_phy)/traj_Tp250_Tt$(Tt)_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	vel_path = "$(data_dir_phy)/vels_Tp250_Tt$(Tt)_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	rho_path = "$(data_dir_phy)/rhos_Tp250_Tt$(Tt)_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	accl = npzread(acc_path);
	traj = npzread(tra_path);
	vels = npzread(vel_path);
	rhos = npzread(rho_path);
	return accl, traj, vels, rhos
end

t_s = 240;
# Vint_gt = obtain_interpolated_velocity_over_t(X_grid, traj_gt, vels_gt, rhos_gt, t_s)
function save_data(method, vel_field)
	println(" ****************** Saving data files for $(method) ***********************")
	vel_path = "./learned_fields/vel_field_$(method)_t$(t_s).npy"
	npzwrite(vel_path, vel_field)
end

# save_data("truth", Vint_gt)


m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi",
          "phys_inf_Wliu_theta_po_liv_Pi"];

m_nn = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		  "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
m_tot = vcat(m_phys, m_nn);

function obtain_itr_lr(method)
	if method =="phys_inf_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="phys_inf_Wab_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="phys_inf_W2ab_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="phys_inf_Wliu_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="node_norm_theta_liv_Pi"
		itr = 500; lr = 0.002;
	end
	if method =="nnsum2_norm_theta_liv_Pi"
		itr = 600; lr = 0.01;
	end
	if method =="rot_inv_theta_liv_Pi"
		itr = 500; lr = 0.02;
	end
	if method =="grad_p_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02;
	end
	if method =="eos_nn_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02;
	end
	return itr, lr
end

for m_ in m_tot
	println("********* running method $(m_) *************")
	itr, lr = obtain_itr_lr(m_);
	accl, traj, vels, rhos = load_sim_coarse_data(m_, "lf", 30, itr, lr);
	Vint = obtain_interpolated_velocity_over_t(X_grid, traj, vels, rhos, t_s)
	save_data(m_, Vint)
end

# itr, lr = obtain_itr_lr(m_tot[6]);
# accl, traj, vels, rhos = load_sim_coarse_data(m_tot[6], "lf", 30, itr, lr);
# Vint = obtain_interpolated_velocity_over_t(X_grid, traj, vels, rhos, t_s)
# save_data(m_tot[6], Vint)
