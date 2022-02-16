using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T = 248;
m_runs = "phys_inf"
l_method = "lf"

function load_losses(method, T)
        Lf = npzread("./learned_generalization/lf_loss_t_$(method)_T$(T).npy")
        Lkl_lf = npzread("./learned_generalization/kl_lf_loss_t_$(method)_T$(T).npy")
        Lkl_t = npzread("./learned_generalization/kl_t_loss_t_$(method)_T$(T).npy")
        Lf_norm = npzread("./learned_generalization/lf_norm_loss_t_$(method)_T$(T).npy")
        return Lf, Lkl_lf, Lkl_t, Lf_norm
end

function obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
    gr(size=(700,600))
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

function obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures/L_kl_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/L_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures/L_lf_norm_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures/L_kl_t_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

    methods = [L"W_{c, \theta}", L"W_{c, \theta; l}", L"W_{c, \theta; \Pi_2}", L"W_{ab, \theta}",
                L"W_{2ab, \theta}", L"W_{q, \theta}", L"W_{c, \theta; p_0}", L"W_{ab; p_0, \theta}"];

    plt_lf = bar(methods, Lf[1,:], yaxis=:log, legend=false)
    title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lf)
    savefig(plt_lf, path_lf)

    plt_lc = bar(methods, Lkl_lf[1,:], yaxis=:log, legend=false)
    title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lc)
    savefig(plt_lc, path_lkl_lf)

    plt_klt = bar(methods, Lkl_t[1,:], yaxis=:log, legend=false)
    title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl}", ytickfontsize=14, yguidefontsize=20)
    display(plt_klt)
    savefig(plt_klt, path_kl_t)

    plt_lfn = bar(methods, Lf_norm[1,:], yaxis=:log, legend=false)
    title!(L"\textrm{Trained Loss Comparison} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end


function plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm)
    t = range(20*dt,T*dt,length=20)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures/L_kl_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/L_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures/L_lf_norm_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures/L_kl_t_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

    methods = [L"W_{c, \theta}" L"W_{c, \theta; l}" L"W_{c, \theta; \Pi_2}" L"W_{ab, \theta}" L"W_{2ab, \theta}" L"W_{q, \theta}" L"W_{c, \theta; p_0}" L"W_{ab; p_0, \theta}"];
    plt_lf = plot(t, Lf, yaxis=:log, linewidth = 2.5, label = methods)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
    savefig(plt_lf, path_lf)

    plt_lc = plot(t, Lkl_lf, yaxis=:log, linewidth = 2.5, label = methods)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
    savefig(plt_lc, path_lkl_lf)

    plt_klt = plot(t, Lkl_t, yaxis=:log, linewidth = 2.5, label = methods)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl}", ytickfontsize=14, yguidefontsize=20)
    savefig(plt_klt, path_kl_t)

    plt_lfn = plot(t, Lf_norm, yaxis=:log, linewidth = 2.5, label = methods)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
    savefig(plt_lfn, path_lf_norm)

end



function obtain_all_losses(all_methods, T)
    Lf_, a,b,c = load_losses("phys_inf_theta", T)
    n_loss = size(Lf_)[1]; n_methods = size(all_methods)[1];
    Lf = zeros(n_loss, n_methods); Lkl_lf = zeros(n_loss, n_methods);
    Lkl_t = zeros(n_loss, n_methods); Lf_norm = zeros(n_loss, n_methods);
    ii = 1;
    for m_ in all_methods
        Lf[:, ii], Lkl_lf[:, ii], Lkl_t[:, ii], Lf_norm[:, ii] = load_losses(m_, T)
        ii += 1;
    end
    return Lf, Lkl_lf, Lkl_t, Lf_norm
end



m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
          "phys_inf_Wliu_theta_po_liv_Pi"];

m_nn = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		  "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
m_tot = vcat(m_phys, m_nn);

methods_phys = [L"W_{c, \theta}" L"W_{c, \theta; l}" L"W_{c, \theta; \Pi_2}" L"W_{ab, \theta}" L"W_{2ab, \theta}" L"W_{q, \theta}" L"W_{c, \theta; p_0}" L"W_{ab; p_0, \theta}"];
methods_phys_bar = [L"W_{c, \theta}", L"W_{c, \theta; l}", L"W_{c, \theta; \Pi_2}", L"W_{ab, \theta}",
            L"W_{2ab, \theta}", L"W_{q, \theta}", L"W_{c, \theta; p_0}", L"W_{ab; p_0, \theta}"];

#
methods_nn = [L"NODE" L"\sum_j NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{ab; p_0, \theta}"];
methods_nn_bar = [L"NODE", L"\sum_j NN", L"Rot-Inv", L"(\nabla P)_{nn}", L"P_{nn}", L"W_{ab; p_0, \theta}"];


Lf, Lkl_lf, Lkl_t, Lf_norm = obtain_all_losses(m_phys, T)

obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm)
obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)



#
# L11, L21, L31, L41 = load_losses(m_phys[1], T)
# L12, L22, L32, L42 = load_losses(m_phys[2], T)
# L13, L23, L33, L43 = load_losses(m_phys[3], T)
# L14, L24, L34, L44 = load_losses(m_phys[4], T)
# L15, L25, L35, L45 = load_losses(m_phys[5], T)
# L16, L26, L36, L46 = load_losses(m_phys[6], T)
# L17, L27, L37, L47 = load_losses(m_phys[7], T)
# L18, L28, L38, L48 = load_losses(m_phys[8], T)
#
# Lf2 = hcat(L11, L12, L13, L14, L15, L16, L17, L18)
#
#
# methods = [L"W_{c, \theta}" L"W_{c, \theta; l}" L"W_{c, \theta; \Pi_2}" L"W_{ab, \theta}" L"W_{2ab, \theta}" L"W_{q, \theta}" L"W_{c, \theta; p_0}" L"W_{ab; p_0, \theta}"];
# #
#
# plt_lf = boxplot(methods, Lf2, yaxis=:log, legend=false, outliers=false)
# title!(L"\textrm{test} ", titlefont=20)
# xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
# ylabel!(L"L_{f}2", ytickfontsize=14, yguidefontsize=20)
# display(plt_lf)


#
# """
# ===============================================================================
# Plotting losses over iterations as a comparison of convergence on trainning set
# ===============================================================================
#
# """
#
# function load_losses_itr(method)
#     dir = "./learned_models/output_data_unif_tracers_forward_$(method)_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height6_m$(method)_lkl_lf_klswitch0"
#     L = npzread("$(dir)/loss.npy")
#     return L
# end
#
#
# #L_wcub_i = load_losses_itr("phys_inf")
# #L_wcub_θ_i = load_losses_itr("phys_inf_theta")
# #L_wliu_i = load_losses_itr("phys_inf_Wliu")
# #L_wab_i = load_losses_itr("phys_inf_Wab")
# #L_w2ab_i = load_losses_itr("phys_inf_W2ab")
# #L_wabp_i = load_losses_itr("phys_inf_Wab_po")
# #L_w2abp_i = load_losses_itr("phys_inf_W2ab_po")
#
# function plot_loss_itr(L_i)
#     gr(size=(700,700))
#     itr = range(0, 2105, length=size(L_i)[1])
#     path_l = "./learned_figures/L_kl_lf_itr.png"
#
#     methods = [L"W_{cub}" L"W_{cub;\theta}" L"W_{quart}" L"W_{ab}" L"W_{ab,po}" L"W_{2ab,po}"];
#     plt_l = plot(itr, L_i, yaxis=:log, linewidth = 2.5, label = methods, palette = :darkrainbow)
#     title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
#     xlabel!(L"itr", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"L_{f} + L_{kl}", ytickfontsize=14, yguidefontsize=20)
#     display(plt_l)
#     savefig(plt_l, path_l)
# end
#
# #Lc_i = hcat(L_wcub_i, L_wcub_θ_i, L_wliu_i, L_wab_i, L_wabp_i, L_w2abp_i);
# #plot_loss_itr(Lc_i)
