using NPZ


m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi",
          "phys_inf_Wliu_theta_po_liv_Pi"];

m_nn = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		  "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi"];
m_tot = vcat(m_phys, m_nn);
# m_all = m_phys
m_all = m_nn


T = 240
l_method = "lf"
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 200;

extern_f = "determistic"
IC = "dns"


θ = 0.0002;
h = 0.335;
t_coarse = 2
dt = t_coarse*0.02;




function save_output_data(data, path)
    npzwrite(path, data)
end

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
		# itr = 500; lr = 0.002; #h=6;
		itr = 600; lr = 0.005;
		# itr = 300; lr = 0.005;
		# itr = 400; lr = 0.005;
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

function load_sim_coarse_data(method, itr, lr)
	data_dir_phy = "./learned_data/"
	acc_path = "$(data_dir_phy)/traj_Tp250_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	tra_path = "$(data_dir_phy)/traj_Tp250_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	vel_path = "$(data_dir_phy)/vels_Tp250_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	rho_path = "$(data_dir_phy)/rhos_Tp250_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
	accl = npzread(acc_path);
	traj = npzread(tra_path);
	vels = npzread(vel_path);
	rhos = npzread(rho_path);
	return accl, traj, vels, rhos
end

# function load_sim_coarse_data(method, itr, lr, Mt)
# 	data_dir_phy = "./learned_data/"
# 	acc_path = "$(data_dir_phy)/traj_Tp30_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
# 	tra_path = "$(data_dir_phy)/traj_Tp30_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
# 	vel_path = "$(data_dir_phy)/vels_Tp30_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
# 	rho_path = "$(data_dir_phy)/rhos_Tp30_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
# 	accl = npzread(acc_path);
# 	traj = npzread(tra_path);
# 	vels = npzread(vel_path);
# 	rhos = npzread(rho_path);
# 	return accl, traj, vels, rhos
# end

include("kde_G.jl")


# mts = [0.04, 0.08, 0.16];
# for Mt in mts
Mt = 0.08;
	# for method in m_all
	method = m_all[1]


	println("*****************    *************")
	println("Running Method = ", method)

	itr, lr = obtain_itr_lr(method)
	accl, traj, vels, rhos = load_sim_coarse_data(method, itr, lr);


	t_inc = T;
	Diff_pred, Vel_inc_pred =
	obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t_inc);
	save_output_data(Vel_inc_pred, "./learned_generalization/vel_inc_pred_$(method)_t$(t_inc)_$(l_method)_Mt$(Mt).npy")
	save_output_data(Diff_pred, "./learned_generalization/diff_pred_$(method)_t$(t_inc)_$(l_method)_Mt$(Mt).npy")

	println("****************** Vel Inc Data Saved for method $(method) ****************")

	# end
# end
