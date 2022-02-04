#

#need to compute for rot_inv, node_norm

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra
using Flux.Losses

m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_Wliu_theta_po_liv_Pi"];

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];

l_m = ["lf", "kl_lf_t"];

# method = m_phys[3];
method = m_nns[5];
l_method = l_m[1];
Mt = 0.08;

ic_method = "unif_tracers"
gsp = 4;
T = 30
t_start = 1;

extern_f = "determistic"
IC = "dns"

h = 0.335;
t_coarse = 2;
dt = 0.02 * t_coarse; c_gt = 0.845; g = 1.4; c = c_gt;

include("./data_loader.jl")
pos_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/pos_traj_4k.npy"
vel_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/vel_traj_4k.npy"
rho_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :];
vels_gt = vels_gt[t_start:t_coarse:end, :, :];
rhos_gt = rhos_gt[t_start:t_coarse:end, :];


D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


# println("*****************    *************")
# println("running method = ", method)
# include("load_models.jl")
# itr = 2200; lr = 0.02; T = 30; θ0 = 0.0009;
# p_fin = load_phys_inf_learned_model(method, l_method, itr, lr, T, θ0, t_coarse, dt)

println("*****************    *************")
println("running method = ", method)
include("load_models.jl")
itr, lr, θ0, class, height = obtain_itr_lr(method)
p_fin, NN, re = load_nn_learned_model(method, l_method, height, class, itr, lr, T, θ0, t_coarse, dt)

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
