using Plots, NPZ, LaTeXStrings, Flux
using Flux.Losses, Statistics

function save_output_data(data, path)
    npzwrite(path, data)
end
function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end


m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];

m_tot = vcat(m_phys, m_nns);

l_method = "lf_klt";
T = 50;
lg_method = "kl_lf";
T_train = 50; T_pred = 500;

loss_method = l_method;
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 1.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 200;

extern_f = "determistic"
IC = "dns"

θ = 0.0002;
h = 0.335;
t_coarse = 1
dt = t_coarse*0.04;


include("./data_loader.jl")
pos_path = "./equil_ic_data/mt008/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt008/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt008/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;


m_n = parse(Int, ARGS[1]);
#for m_n in 5 : 9
	method = m_tot[m_n];
	println("*****************    *************")
	println("Running Method = ", method)

	function load_sim_coarse_data(method)
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

	accl, traj, vels, rhos = load_sim_coarse_data(method);


	include("./loss_functions.jl")
	make_dir("learned_field_data_t50_lf_klt");
	# Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(X_grid, traj_gt, vels_gt, rhos_gt, T_pred, N_f)
	function obtain_vel_field_values(T_pred)
		Vf_pr,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(X_grid, traj, vels, rhos, T_pred, N_f)
		return Vf_pr
	end

	println("*************** Obtaining Velocity Field Values ************")
	Vf_pr = obtain_vel_field_values(T_pred)


	println("*************** Velocity Field obtained ************")

	# save_output_data(Vf_gt, "./learned_field_data_t50/Vf_gt_Tp500.npy")
	save_output_data(Vf_pr, "./learned_field_data_t50_lf_klt/Vf_pr_$(method)_$(l_method)_Tp$(T_pred)_Tt$(T_train).npy")
#end
