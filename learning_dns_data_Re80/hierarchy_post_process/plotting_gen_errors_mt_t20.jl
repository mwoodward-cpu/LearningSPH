using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T_pred = 500;
T_train = 20;

# m_runs = "phys_inf"
# m_runs = "nns"
# m_runs = "comb"
l_method = "lf"
n_loss = 3 #max 20;
include("./color_scheme_utils.jl")
cs1=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
cs2=[c6,c7,c8,c9];
cs3=[c1,c2,c3,c4,c5];
if m_runs == "comp" cs = cs1; end
if m_runs == "phys_inf" cs = cs2; end
if m_runs == "nns" cs = cs3; end


function load_losses_mt(method, T_pred, T_train)
        Lf = npzread("./learned_generalization_mt_rel/lf_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        # Lkl_lf = npzread("./learned_generalization_mt_rel/kl_lf_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        Lkl_t = npzread("./learned_generalization_mt_rel/kl_t_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        Lf_norm = npzread("./learned_generalization_mt_rel/lf_norm_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        return Lf[1:n_loss], Lkl_t[1:n_loss], Lf_norm[1:n_loss]
end



function obtain_box_plot_t_lf_comb(methods, Lf, Lf_norm)
    # gr(size=(900,750))

    path_lf = "./learned_figures_mt_rel/L_lf_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_mt_rel/L_lf_norm_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"


    plt_lfn = boxplot([methods[1]], Lf_norm[:, 1], color = cs[1], legend=false, outliers=false)
	boxplot!([methods[2]], Lf_norm[:, 2], color = cs[2], legend=false, outliers=false)
	boxplot!([methods[3]], Lf_norm[:, 3], color = cs[3], legend=false, outliers=false)
	boxplot!([methods[4]], Lf_norm[:, 4], color = cs[4], legend=false, outliers=false)
	boxplot!([methods[5]], Lf_norm[:, 5], color = cs[5], legend=false, outliers=false)
	boxplot!([methods[6]], Lf_norm[:, 6], color = cs[6], legend=false, outliers=false)
	boxplot!([methods[7]], Lf_norm[:, 7], color = cs[7], legend=false, outliers=false)
	boxplot!([methods[8]], Lf_norm[:, 8], color = cs[8], legend=false, outliers=false)
	boxplot!([methods[9]], Lf_norm[:, 9], color = cs[9], legend=false, outliers=false)

	# dotplot!(methods, Lf_norm, marker=(:black, stroke(0)));
    title!(L"\textrm{Generalization ~ Error ~ Over: ~ } M_t ", titlefont=title_fs)
    xlabel!(L"\textrm{Method}", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
    ylabel!(L"L_{f}/\left<ke\right>", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end


function obtain_box_plot_t_lf_phys(methods, Lf, Lf_norm)
    # gr(size=(900,750))

    path_lf = "./learned_figures_mt_rel/L_lf_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_mt_rel/L_lf_norm_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"


    plt_lfn = boxplot([methods[1]], Lf_norm[:, 1], color = cs[1], legend=false, outliers=false)
	boxplot!([methods[2]], Lf_norm[:, 2], color = cs[2], legend=false, outliers=false)
	boxplot!([methods[3]], Lf_norm[:, 3], color = cs[3], legend=false, outliers=false)
	boxplot!([methods[4]], Lf_norm[:, 4], color = cs[4], legend=false, outliers=false)

	# dotplot!(methods, Lf_norm, marker=(:black, stroke(0)));
    title!(L"\textrm{Generalization ~ Error ~ Over: ~ } M_t ", titlefont=title_fs)
    xlabel!(L"\textrm{Method}", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
    ylabel!(L"L_{f}/\left<ke\right>", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end



function obtain_all_losses(all_methods, T_pred, T_train)
    # Lf_, a,b,c = load_losses("phys_inf_W2ab_theta_po_liv_Pi", T_pred, T_train)
    n_methods = size(all_methods)[1];
    Lf = zeros(n_loss, n_methods); Lkl_lf = zeros(n_loss, n_methods);
    Lkl_t = zeros(n_loss, n_methods); Lf_norm = zeros(n_loss, n_methods);
    ii = 1;
    for m_ in all_methods
        Lf[:, ii], Lkl_t[:, ii], Lf_norm[:, ii] = load_losses_mt(m_, T_pred, T_train)
        ii += 1;
    end
    return Lf, Lkl_t, Lf_norm
end



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

T_pred = 500; T_train = 20;


if m_runs=="phys_inf"
	Lf, Lkl_t, Lf_norm = obtain_all_losses(m_phys, T_pred, T_train)
	obtain_box_plot_t_lf_phys(methods_phys, Lf, Lf_norm)
end

if m_runs=="nns"
	Lf, Lkl_t, Lf_norm = obtain_all_losses(m_nns, T_pred, T_train)
	obtain_box_plot_t_lf(methods_nn, Lf, Lf_norm)
end

if m_runs=="comb"
	Lf, Lkl_t, Lf_norm = obtain_all_losses(m_comb, T_pred, T_train)
	obtain_box_plot_t_lf_comb(methods_comb, Lf, Lf_norm)
end


# obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
# plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm)
# obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
