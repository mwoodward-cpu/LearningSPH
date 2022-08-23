#

#need to compute for rot_inv, node_norm

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra
using Flux.Losses

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];

# class = "phys"
class = "nns"

include("./load_models_t20.jl")
T_train = 20; l_method = "lf"
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()
phys_dirs = [d1, d2, d3, d4];
nn_dirs = [d5, d6, d7, d8, d9];

Mt = 0.08
mmt = "008";
h=0.335
t_start = 1;
t_coarse=1;

include("./data_loader.jl")
pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;

extern_f="determistic"
IC = "dns_equil"

# for m_num_nn in 1 : 5
m_num_nn = 5;

	if class=="phys"
		pdir = phys_dirs[m_num];
		method = m_phys[m_num];
	end

	if class=="nns"
		pdir = nn_dirs[m_num_nn];
		method = m_nns[m_num_nn];
	end


	l_m = ["lf", "kl_lf_t"];
	l_method = l_m[1];

	gsp = 4;
	T = T_train

	h = 0.335;
	t_coarse = 1;
	dt = 0.04 * t_coarse;


	println("*****************    *************")
	println("running method = ", method)
	if pdir in phys_dirs
		p_fin = load_learned_model_params(pdir)
	end

	if pdir in nn_dirs
		p_fin, NN, re = load_nn_learned_model(pdir)
		height = 6;
		if m_num_nn==1 || m_num_nn==4
			height = 8
		end
		n_params = size(p_fin)[1];
	end

	function include_sensitivity_file(method)
		include("./models/sensitivities_3d_$(method).jl")
	end

	if method!="truth"
		include_sensitivity_file(method)
		include("./integrators_utils.jl")
		rQ_, rR = rotational_metric(p_fin, obtain_sph_AV_A)
		g_, gal_inv = translational_metric(p_fin, obtain_sph_AV_A)
		println("rot error = ", rR); println("gal_inv = ", gal_inv);
	end

	rR_path = "./learned_data/rR_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(Mt).npy"
	gal_path = "./learned_data/gal_inv_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(Mt).npy"
	trans_path = "./learned_data/tra_inv_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(Mt).npy"

	function gen_data_files(rR, gal_inv)
	    println(" ****************** Saving data files ***********************")
		npzwrite(rR_path, rR)
		npzwrite(gal_path, gal_inv)
	end

	gen_data_files(rR, gal_inv)

# end
