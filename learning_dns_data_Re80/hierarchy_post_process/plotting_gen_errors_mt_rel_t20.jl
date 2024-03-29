using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T_pred = 500;
T_train = 20; T = T_pred

m_runs = "phys_inf"
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


function load_losses(method, T_pred, T_train)
        Lf = npzread("./learned_generalization_mt_rel/lf_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        # Lkl_lf = npzread("./learned_generalization_mt_rel/kl_lf_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        Lkl_t = npzread("./learned_generalization_mt_rel/kl_t_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        Lf_norm = npzread("./learned_generalization_mt_rel/lf_norm_loss_t_$(method)_T$(T_train)_over_Mt.npy")
        return Lf[1:n_loss], Lkl_t[1:n_loss], Lf_norm[1:n_loss]
end


function obtain_box_plot_t_lf_comb(methods, Lf, Lf_norm)
    gr(size=(900,750))

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
    title!(L"\textrm{Generalization ~ Error ~ Over: ~ } M_t ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=16, xguidefontsize=20)
    ylabel!(L"L_{f}/\left<ke\right>", ytickfontsize=16, yguidefontsize=20)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end



function obtain_box_plot_t_lf_nns(methods, Lf, Lf_norm)
    gr(size=(900,750))

    path_lf = "./learned_figures_mt_rel/L_lf_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_mt_rel/L_lf_norm_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"


    plt_lfn = boxplot([methods[1]], Lf_norm[:, 1], color = cs[1], legend=false, outliers=false)
	boxplot!([methods[2]], Lf_norm[:, 2], color = cs[2], legend=false, outliers=false)
	boxplot!([methods[3]], Lf_norm[:, 3], color = cs[3], legend=false, outliers=false)
	boxplot!([methods[4]], Lf_norm[:, 4], color = cs[4], legend=false, outliers=false)
	boxplot!([methods[5]], Lf_norm[:, 5], color = cs[5], legend=false, outliers=false)

	# dotplot!(methods, Lf_norm, marker=(:black, stroke(0)));
    title!(L"\textrm{Generalization ~ Error ~ Over: ~ } M_t ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=16, xguidefontsize=20)
    ylabel!(L"L_{f}/\left<ke\right>", ytickfontsize=16, yguidefontsize=20)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end

function obtain_box_plot_t_lf_phys(methods, Lf, Lf_norm)
    gr(size=(900,750))

    path_lf = "./learned_figures_mt_rel/L_lf_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_mt_rel/L_lf_norm_box_plot_over_Mt_T$(T)_$(m_runs)_$(l_method).png"


    plt_lfn = boxplot([methods[1]], Lf_norm[:, 1], color = cs[1], legend=false, outliers=false)
	boxplot!([methods[2]], Lf_norm[:, 2], color = cs[2], legend=false, outliers=false)
	boxplot!([methods[3]], Lf_norm[:, 3], color = cs[3], legend=false, outliers=false)
	boxplot!([methods[4]], Lf_norm[:, 4], color = cs[4], legend=false, outliers=false)

	# dotplot!(methods, Lf_norm, marker=(:black, stroke(0)));
    title!(L"\textrm{Generalization ~ Error ~ Over: ~ } M_t", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=16, xguidefontsize=22)
    ylabel!(L"L_{f}/\left<ke\right>", ytickfontsize=16, yguidefontsize=22)
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
        Lf[:, ii], Lkl_t[:, ii], Lf_norm[:, ii] = load_losses(m_, T_pred, T_train)
        ii += 1;
    end
    return Lf, Lkl_lf, Lkl_t, Lf_norm
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

T_pred = 200; T_train = 20;

if m_runs=="phys_inf"
	Lf, Lkl_lf, Lkl_t, Lf_norm = obtain_all_losses(m_phys, T_pred, T_train)
	obtain_box_plot_t_lf_phys(methods_phys, Lf, Lf_norm)
end

if m_runs=="nns"
	Lf, Lkl_lf, Lkl_t, Lf_norm = obtain_all_losses(m_nns, T_pred, T_train)
	obtain_box_plot_t_lf_nns(methods_nn, Lf, Lf_norm)
end

if m_runs=="comb"
	Lf, Lkl_lf, Lkl_t, Lf_norm = obtain_all_losses(m_comb, T_pred, T_train)
	obtain_box_plot_t_lf_comb(methods_comb, Lf, Lf_norm)
end

# obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
# plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm)
# obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



function obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
    gr(size=(700,600))
	c1 = "blue2"; c2 = "darkorange1"; c3 = "darkgreen"; c4 = "purple3"; c5 = "springgreen2";
	c6 = "tan"; c7 = "deepskyblue3"; c8 = "yellow3"; c9 = "red3";
	cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9];

    path_lkl_lf = "./learned_figures/L_kl_lf_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/L_lf_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures/L_lf_norm_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures/L_kl_t_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

    methods = [L"W_{c, \theta}" L"W_{c, \theta; l}" L"W_{c, \theta; \Pi_2}" L"W_{ab, \theta}" L"W_{2ab, \theta}" L"W_{q, \theta}" L"W_{c, \theta; p_0}" L"W_{ab; p_0, \theta}"];

    plt_lf = boxplot(methods, Lf, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lf)
    savefig(plt_lf, path_lf)

    plt_lc = boxplot(methods, Lkl_lf, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lc)
    savefig(plt_lc, path_lkl_lf)

    plt_klt = boxplot(methods, Lkl_t, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl}", ytickfontsize=14, yguidefontsize=20)
    display(plt_klt)
    savefig(plt_klt, path_kl_t)

    plt_lfn = boxplot(methods, Lf_norm, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end

# function obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
#     gr(size=(700,600))
#     path_lkl_lf = "./learned_figures/L_kl_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#     path_lf = "./learned_figures/L_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#     path_lf_norm = "./learned_figures/L_lf_norm_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#     path_kl_t = "./learned_figures/L_kl_t_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#
#     methods = [L"W_{c, \theta}", L"W_{c, \theta; l}", L"W_{c, \theta; \Pi_2}", L"W_{ab, \theta}",
#                 L"W_{2ab, \theta}", L"W_{q, \theta}", L"W_{c, \theta; p_0}", L"W_{ab; p_0, \theta}"];
#
#     plt_lf = bar(methods, Lf[1,:], yaxis=:log, legend=false)
#     title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
#     display(plt_lf)
#     savefig(plt_lf, path_lf)
#
#     plt_lc = bar(methods, Lkl_lf[1,:], yaxis=:log, legend=false)
#     title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
#     display(plt_lc)
#     savefig(plt_lc, path_lkl_lf)
#
#     plt_klt = bar(methods, Lkl_t[1,:], yaxis=:log, legend=false)
#     title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{kl}", ytickfontsize=14, yguidefontsize=20)
#     display(plt_klt)
#     savefig(plt_klt, path_kl_t)
#
#     plt_lfn = bar(methods, Lf_norm[1,:], yaxis=:log, legend=false)
#     title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
#     display(plt_lfn)
#     savefig(plt_lfn, path_lf_norm)
# end
#
#
# function plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm)
#     t = range(20*dt,T*dt,length=20)
#     gr(size=(700,600))
#     path_lkl_lf = "./learned_figures/L_kl_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#     path_lf = "./learned_figures/L_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#     path_lf_norm = "./learned_figures/L_lf_norm_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#     path_kl_t = "./learned_figures/L_kl_t_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
#
#     methods = [L"W_{c, \theta}" L"W_{c, \theta; l}" L"W_{c, \theta; \Pi_2}" L"W_{ab, \theta}" L"W_{2ab, \theta}" L"W_{q, \theta}" L"W_{c, \theta; p_0}" L"W_{ab; p_0, \theta}"];
#     plt_lf = plot(t, Lf, yaxis=:log, linewidth = 2.5, label = methods)
#     title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
#     xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
#     savefig(plt_lf, path_lf)
#
#     plt_lc = plot(t, Lkl_lf, yaxis=:log, linewidth = 2.5, label = methods)
#     title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
#     xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
#     savefig(plt_lc, path_lkl_lf)
#
#     plt_klt = plot(t, Lkl_t, yaxis=:log, linewidth = 2.5, label = methods)
#     title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
#     xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{kl}", ytickfontsize=14, yguidefontsize=20)
#     savefig(plt_klt, path_kl_t)
#
#     plt_lfn = plot(t, Lf_norm, yaxis=:log, linewidth = 2.5, label = methods)
#     title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
#     xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
#     savefig(plt_lfn, path_lf_norm)
#
# end
