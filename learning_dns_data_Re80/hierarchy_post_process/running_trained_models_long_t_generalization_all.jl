"""
Using Trained models to make long time predcitions (9 times longer than training)

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];

# method = m_phys[1];

function load_dirs_names_t20()
	dir_t20 = "learned_models_t20_minl"
	d1 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_W2ab_theta_po_liv_Pi_lf_skl0_itr2300_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_W2ab_theta_po_liv_Pi_llf_klswitch0"
	d2 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wab_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wab_theta_po_liv_Pi_llf_klswitch0"
	d3 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wliu_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wliu_theta_po_liv_Pi_llf_klswitch0"
	d4 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mphys_inf_theta_po_liv_Pi_llf_klswitch0"
	d5 = "$(dir_t20)/output_data_unif_tracers_forward_node_norm_theta_liv_Pi_lf_itr800_lr0.002_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height8_mnode_norm_theta_liv_Pi_llf_klswitch0"
	d6 = "$(dir_t20)/output_data_unif_tracers_forward_nnsum2_norm_theta_liv_Pi_lf_itr600_lr0.01_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mnnsum2_norm_theta_liv_Pi_llf_klswitch0"
	d7 = "$(dir_t20)/output_data_unif_tracers_forward_rot_inv_theta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mrot_inv_theta_liv_Pi_llf_klswitch0"
	d8 = "$(dir_t20)/output_data_unif_tracers_forward_eos_nn_theta_alpha_beta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height8_meos_nn_theta_alpha_beta_liv_Pi_llf_klswitch0"
	d9 = "$(dir_t20)/output_data_unif_tracers_forward_grad_p_theta_alpha_beta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mgrad_p_theta_alpha_beta_liv_Pi_llf_klswitch0"
	return d1, d2, d3, d4, d5, d6, d7, d8, d9
end

T_train = 20; l_method = "lf"
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()
phys_dirs = [d1, d2, d3, d4];
# pdir = phys_dirs[1];

for i in 2 : 4
	pdir = phys_dirs[i];
	method = m_phys[i];

	include("./utils.jl")
	make_dir("learned_figures"); make_dir("learned_sims")
	make_dir("learned_data")

	T_pred = 500
	t_start = 1;

	extern_f = "determistic"
	IC = "dns_equil"

	h = 0.335;
	t_coarse = 1;
	dt = 0.04 * t_coarse;

	include("./data_loader.jl")
	pos_path = "./equil_ic_data/mt008/pos_traj_4k_unif.npy"
	vel_path = "./equil_ic_data/mt008/vel_traj_4k_unif.npy"
	rho_path = "./equil_ic_data/mt008/rho_traj_4k_unif.npy"
	traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

	traj_gt = traj_gt[t_start:t_coarse:end, :, :];
	vels_gt = vels_gt[t_start:t_coarse:end, :, :];
	rhos_gt = rhos_gt[t_start:t_coarse:end, :];


	D = size(traj_gt)[3];
	N = size(traj_gt)[2];
	m = (2.0 * pi)^D / N;


	println("*****************    *************")
	println("running method = ", method)
	include("load_models_t20.jl")
	p_fin = load_learned_model_params(pdir)


	function include_sensitivity_file(method)
		include("./models/sensitivities_3d_$(method).jl")
	end

	if method!="truth"
		include_sensitivity_file(method)
		include("./sph_3d_integrator.jl")
		accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, T_pred)
	end




	#-----------Outputs

	function simulate(pos, sim_time=15)
	    sim_path = "./learned_sims/traj_N$(N)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method).mp4"
	    gr(size=(800,700))
	    println("**************** Simulating the particle flow ***************")
	    #theme(:juno)
	    n_2 = round(Int,N/2); ms_ = 7.0;
	    anim = @animate for i ∈ 1 : T_pred
			println("time step = ", i)
	         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
	         title = "Learned WCSPH_$(method): N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
	         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], ms=ms_, color = "red")
	    end
	    gif(anim, sim_path, fps = round(Int, T_pred/sim_time))
	    println("****************  Simulation COMPLETE  *************")
	end


	acc_path = "./learned_data/accl_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	pos_path = "./learned_data/traj_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	vel_path = "./learned_data/vels_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	rho_path = "./learned_data/rhos_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"

	function gen_data_files(accl, traj, vels, rhos)
	    println(" ****************** Saving data files ***********************")
		# npzwrite(vor_path, vort[t_save:end,:,:])
		npzwrite(acc_path, accl)
		npzwrite(pos_path, traj)
		npzwrite(vel_path, vels)
		npzwrite(rho_path, rhos)
	end

	gen_data_files(accl, traj, vels, rhos)

	#UAHPC seems to need this formatting
	ENV["GKSwstype"]="100"
	simulate(traj, 15)
	if method=="truth"
		simulate(traj_gt, 15)
	end
end

method="truth"
if method=="truth"
	simulate(traj_gt, 15)
end
