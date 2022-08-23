using NPZ


m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
# m_tot = vcat(m_phys, m_nn);
m_comb = vcat(m_nns, m_phys)

methods_phys = [L"W_{2}(a,b)" L"W(a,b)" L"W_{cub}" L"W_{quart}"];

#
methods_nn = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}"] # L"W_{ab; p_0, \theta}"];
methods_comb = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{cub}" L"W_{2}(a,b)"] # L"W_{ab; p_0, \theta}"];

m_all = vcat(m_nns, m_phys);



# Mt = 0.16; mmt = "016";
# Mt = 0.04; mmt = "004";
Mt = 0.08; mmt = "008"


T_pred = 500; T_train = 50;
l_method = "lf_klt"
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 200;

extern_f = "determistic"
IC = "dns_equil"


θ = 0.0002;
h = 0.335;
t_coarse = 1
dt = t_coarse*0.04;




function save_output_data(data, path)
    npzwrite(path, data)
end

include("./data_loader.jl")
pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;


function load_data(method, IC, T_pred, T_train)
	acc_path = "./learned_data_t50_lf_klt/accl_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	tra_path = "./learned_data_t50_lf_klt/traj_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	vel_path = "./learned_data_t50_lf_klt/vels_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	rho_path = "./learned_data_t50_lf_klt/rhos_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	accl = npzread(acc_path);
	traj = npzread(tra_path);
	vels = npzread(vel_path);
	rhos = npzread(rho_path);
	return accl, traj, vels, rhos
end

function load_data_mt(method, IC, T_pred, T_train, mt)
	acc_path = "./learned_data_t50_lfklt/accl_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
	tra_path = "./learned_data_t50_lfklt/traj_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
	vel_path = "./learned_data_t50_lfklt/vels_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
	rho_path = "./learned_data_t50_lfklt/rhos_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
	accl = npzread(acc_path);
	traj = npzread(tra_path);
	vels = npzread(vel_path);
	rhos = npzread(rho_path);
	return accl, traj, vels, rhos
end

include("kde_G.jl")


# mts = [0.04, 0.08, 0.16];
# for Mt in mts
	for m_n in 1 : 9
	if m_n < 6 IC = "dns" end
	if m_n > 5 IC = "dns_equil" end
	method = m_all[m_n]

	println("*****************    *************")
	println("Running Method = ", method)
	if Mt==0.08
		accl, traj, vels, rhos = load_data(method, IC, T_pred, T_train);
	else
		accl, traj, vels, rhos = load_data_mt(method, IC, T_pred, T_train, Mt)
	end

	if Mt ==0.08
		t_inc = T_pred;
	else
		t_inc = T_train;
	end
	Diff_pred, Vel_inc_pred =
	obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t_inc);
	save_output_data(Vel_inc_pred, "./learned_generalization_t50_lfklt/vel_inc_pred_$(method)_t$(t_inc)_tp$(T_pred)_$(l_method)_Mt$(Mt).npy")
	save_output_data(Diff_pred, "./learned_generalization_t50_lfklt/diff_pred_$(method)_t$(t_inc)_tp$(T_pred)_$(l_method)_Mt$(Mt).npy")

	println("****************** Vel Inc Data Saved for method $(method) ****************")

	end
# end
