using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T_pred = 500;
T_train = 50;

# m_runs = "phys_inf"
# m_runs = "nns"
m_runs = "comb"
l_method = "lf_klt"
n_loss = 18 #max 20;

function load_losses(method, T_pred, T_train)
        # Lf = npzread("./learned_generalization_t50/lf_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train).npy")
        # Lkl_lf = npzread("./learned_generalization_t50/kl_lf_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train).npy")
        # Lkl_t = npzread("./learned_generalization_t50/kl_t_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train)_over_t.npy")
        Lf_norm = npzread("./learned_generalization_t50/lf_norm_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train)_over_t.npy")
        return Lf_norm[1:n_loss]
end

function obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures_t50/L_kl_lf_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures_t50/L_lf_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_t50/L_lf_norm_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures_t50/L_kl_t_box_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

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

function obtain_box_plot_t_lf(methods, Lf_norm)
    gr(size=(700,600))
    path_lf = "./learned_figures_t50/L_lf_box_plot_over_t_T$(T_pred)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_t50/L_lf_norm_box_plot_over_t_T$(T_pred)_$(m_runs)_$(l_method).png"

    # plt_lf = boxplot(methods, Lf, legend=false, outliers=false)
    # title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    # xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    # ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
    # display(plt_lf)
    # savefig(plt_lf, path_lf)

    plt_lfn = boxplot(methods, Lf_norm, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{f}/\left<ke\right>", ytickfontsize=14, yguidefontsize=20)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)
end

function obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures_t50/L_kl_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures_t50/L_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_t50/L_lf_norm_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures_t50/L_kl_t_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

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
    path_lkl_lf = "./learned_figures_t50/L_kl_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures_t50/L_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures_t50/L_lf_norm_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures_t50/L_kl_t_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

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



function obtain_all_losses(all_methods, T_pred, T_train)
    # Lf_, a,b,c = load_losses("phys_inf_W2ab_theta_po_liv_Pi", T_pred, T_train)
    n_methods = size(all_methods)[1];
    Lf_norm = zeros(n_loss, n_methods);
    ii = 1;
    for m_ in all_methods
        Lf_norm[:, ii] = load_losses(m_, T_pred, T_train)
        ii += 1;
    end
    return Lf_norm
end



m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
# m_tot = vcat(m_phys, m_nn);
m_comb = vcat(m_nns[1], m_nns[3:5], m_phys[4], m_phys[1])

methods_phys = [L"W_{2}(a,b)" L"W(a,b)" L"W_{cub}" L"W_{quart}"];


#
methods_nn = [L"NODE" L"\sum_j NN" L"Rot-Inv" L"P_{nn}" L"(\nabla P)_{nn}"] # L"W_{ab; p_0, \theta}"];
methods_nn_bar = [L"NODE", L"\sum_j NN", L"Rot-Inv", L"P_{nn}", L"(\nabla P)_{nn}"] #L"W_{ab; p_0, \theta}"];

methods_comb = [L"NODE" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{cub}" L"W_{2}(a,b)"] # L"W_{ab; p_0, \theta}"];

T_pred = 500; T_train = 50;

if m_runs=="phys_inf"
	Lf_norm = obtain_all_losses(m_phys, T_pred, T_train)
	obtain_box_plot_t_lf(methods_phys, Lf_norm)
end

if m_runs=="nns"
	Lf_norm = obtain_all_losses(m_nns, T_pred, T_train)
	obtain_box_plot_t_lf(methods_nn, Lf_norm)
end

if m_runs=="comb"
	Lf_norm = obtain_all_losses(m_comb, T_pred, T_train)
	obtain_box_plot_t_lf(methods_comb, Lf_norm)
end
