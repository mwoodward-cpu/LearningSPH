using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T = 30;
Mt = 0.08;

m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wliu_theta_po_liv_Pi",
		  "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
# m_tot = vcat(m_phys, m_nn);
m_comb = vcat(m_nns, m_phys)

methods_phys = [L"W_{cub}" L"W_{quart}" L"W(a,b)" L"W_{2}(a,b)"];


#
methods_nn = [L"NODE" L"\sum NN" L"RotInv" L"(\nabla P)_{nn}" L"P_{nn}"] # L"W_{ab; p_0, \theta}"];
methods_comb = [L"NODE" L"\sum NN" L"RotInv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{cub}" L"W_{quart}" L"W(a,b)" L"W_{2}(a,b)"] # L"W_{ab; p_0, \theta}"];

Mt = 0.08;

T = 20;
rf_node = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_nns[1])_lf_mt$(Mt).npy")[end]
gf_node = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_nns[1])_lf_mt$(Mt).npy")[end]

rf_nnsm = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_nns[2])_lf_mt$(Mt).npy")[end]
gf_nnsm = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_nns[2])_lf_mt$(Mt).npy")[end]

rf_rotI = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_nns[3])_lf_mt$(Mt).npy")[end]
gf_rotI = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_nns[3])_lf_mt$(Mt).npy")[end]

rf_grap = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_nns[4])_lf_mt$(Mt).npy")[end]
gf_grap = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_nns[4])_lf_mt$(Mt).npy")[end]

rf_eosn = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_nns[5])_lf_mt$(Mt).npy")[end]
gf_eosn = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_nns[5])_lf_mt$(Mt).npy")[end]


rf_wcub = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_comb[6])_lf_mt$(Mt).npy")[end]
gf_wcub = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_comb[6])_lf_mt$(Mt).npy")[end]

rf_wliu = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_comb[7])_lf_mt$(Mt).npy")[end]
gf_wliu = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_comb[7])_lf_mt$(Mt).npy")[end]

rf_wab = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_comb[8])_lf_mt$(Mt).npy")[end]
gf_wab = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_comb[8])_lf_mt$(Mt).npy")[end]

rf_w2ab = npzread("./learned_data/rR_T$(T)_h0.335_dns_equil_$(m_comb[9])_lf_mt$(Mt).npy")[end]
gf_w2ab = npzread("./learned_data/gal_inv_T$(T)_h0.335_dns_equil_$(m_comb[9])_lf_mt$(Mt).npy")[end]






RF = vcat(rf_node, rf_nnsm, rf_rotI, rf_grap, rf_eosn, rf_wcub, rf_wliu, rf_wab, rf_w2ab)
GF = vcat(gf_node, gf_nnsm, gf_rotI, gf_grap, gf_eosn, gf_wcub, gf_wliu, gf_wab, gf_w2ab)

function obtain_bar_plot_symm(RF, GF, methods)
    # gr(size=(700,600))
    path_lkl_lf = "./learned_figures/RF_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/GF_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

    plt_lf = bar([methods[1]], [RF[1]], yaxis=:log, color = cs[1], legend=false)
	bar!([methods[2]], [RF[2]], yaxis=:log, color = cs[2], legend=false)
	bar!([methods[3]], [RF[3]], yaxis=:log, color = cs[3], legend=false)
	bar!([methods[4]], [RF[4]], yaxis=:log, color = cs[4], legend=false)
	bar!([methods[5]], [RF[5]], yaxis=:log, color = cs[5], legend=false)
	bar!([methods[6]], [RF[6]], yaxis=:log, color = cs[6], legend=false)
	bar!([methods[7]], [RF[7]], yaxis=:log, color = cs[7], legend=false)
	bar!([methods[8]], [RF[8]], yaxis=:log, color = cs[8], legend=false)
	bar!([methods[9]], [RF[9]], yaxis=:log, color = cs[9], legend=false)
    title!(L"\textrm{Rotational ~ Invariance ~ Error} ", titlefont=title_fs)
    xlabel!(L"\textrm{Method}", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
    ylabel!(L"||R{F}(X) - {F}(RX)||_2/||{F}(X)||_2", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)
    display(plt_lf)
    savefig(plt_lf, path_lf)

    plt_lc = bar(methods, GF, yaxis=:log, color=[csn[i] for i in 1:9], legend=false)
    title!(L"\textrm{Galilean ~ Invariance ~ Error} ", titlefont=title_fs)
    xlabel!(L"\textrm{Method}", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
    ylabel!(L"||{F}(V) - {F}(V - V')||_2/||{F}(V)||_2", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)
    display(plt_lc)
    savefig(plt_lc, path_lkl_lf)
end

obtain_bar_plot_symm(RF, GF, methods_comb[1,:])
