using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T = 30;
Mt = 0.08;

# m_nn_comp = ["node_norm", "nnsum2_norm_theta", "rot_inv", "grad_p_theta", "eos_nn_theta", "phys_inf_Wab_po_theta"];
methods_nn = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{ab; p_0, \theta}"];
methods_nn_bar = [L"NODE", L"\sum NN", L"Rot-Inv", L"(\nabla P)_{nn}", L"P_{nn}", L"W_{ab; p_0, \theta}"];
m_nn = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi"];

#
include("./data_loader.jl")
pos_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/pos_traj_4k.npy"
vel_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/vel_traj_4k.npy"
rho_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :];
vels_gt = vels_gt[t_start:t_coarse:end, :, :];
rhos_gt = rhos_gt[t_start:t_coarse:end, :];


D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;

function compute_gt_accl(vel_gt)
	accl_gt = zeros(T, N, D);
	for t in 1 : (T-1)
		for i in 1 : D
			accl_gt[t+1,:,i] = (vel_gt[t+1,:,i] .- vel_gt[t,:,i]) / dt
		end
	end
	return accl_gt
end

accl_gt = compute_gt_accl(vels_gt)

m_runs = "nn_comp"
l_method = "lf"

rf_node = npzread("./learned_data/rR_T$(T)_h0.335_dns_$(m_nn[1])_lf_mt$(Mt).npy")[end]
gf_node = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_$(m_nn[1])_lf_mt$(Mt).npy")[end]

rf_nnsm = npzread("./learned_data/rR_T$(T)_h0.335_dns_$(m_nn[2])_lf_mt$(Mt).npy")[end]
gf_nnsm = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_$(m_nn[2])_lf_mt$(Mt).npy")[end]

rf_rotI = npzread("./learned_data/rR_T$(T)_h0.335_dns_$(m_nn[3])_lf_mt$(Mt).npy")[end]
gf_rotI = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_$(m_nn[3])_lf_mt$(Mt).npy")[end]

rf_grap = npzread("./learned_data/rR_T$(T)_h0.335_dns_$(m_nn[4])_lf_mt$(Mt).npy")[end]
gf_grap = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_$(m_nn[4])_lf_mt$(Mt).npy")[end]

rf_eosn = npzread("./learned_data/rR_T$(T)_h0.335_dns_$(m_nn[5])_lf_mt$(Mt).npy")[end]
gf_eosn = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_$(m_nn[5])_lf_mt$(Mt).npy")[end]

rf_wabp = npzread("./learned_data/rR_T$(T)_h0.335_dns_$(m_nn[6])_lf_mt$(Mt).npy")[end]
gf_wabp = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_$(m_nn[6])_lf_mt$(Mt).npy")[end]


RF = vcat(rf_node, rf_nnsm, rf_rotI, rf_grap, rf_eosn, rf_wabp)
GF = vcat(gf_node, gf_nnsm, gf_rotI, gf_grap, gf_eosn, gf_wabp)

function obtain_bar_plot_symm(RF, GF, methods)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures/RF_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/GF_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

    plt_lf = bar(methods, RF, yaxis=:log, legend=false)
    title!(L"\textrm{Rotational Invariance Error} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"||R{F}(X) - {F}(RX)||_2/||{F}(X)||_2", ytickfontsize=14, yguidefontsize=20)
    display(plt_lf)
    savefig(plt_lf, path_lf)

    plt_lc = bar(methods, GF, yaxis=:log, legend=false)
    title!(L"\textrm{Galilean Invariance Error} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"||{F}(V) - {F}(V - V')||_2/||{F}(V)||_2", ytickfontsize=14, yguidefontsize=20)
    display(plt_lc)
    savefig(plt_lc, path_lkl_lf)
end

obtain_bar_plot_symm(RF, GF, methods_nn_bar)
