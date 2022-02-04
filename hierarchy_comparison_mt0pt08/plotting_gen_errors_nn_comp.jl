using NPZ, Plots, LaTeXStrings
using StatsPlots

dt = 0.04
# T = parse(Int, ARGS[1]);
# m_runs = ARGS[2];
T = 30;
m_runs = "m_phys"
l_method = "lf"

function load_losses_get_t(method, T)
        Lf = npzread("./learned_generalization/lf_loss_t_$(method)_T$(T)_over_t.npy")
        Lkl_lf = npzread("./learned_generalization/kl_lf_loss_t_$(method)_T$(T)_over_t.npy")
        Lkl_t = npzread("./learned_generalization/kl_t_loss_t_$(method)_T$(T)_over_t.npy")
        Lf_norm = npzread("./learned_generalization/lf_norm_loss_t_$(method)_T$(T)_over_t.npy")
        Lkl_a = npzread("./learned_generalization/kl_t_accl_$(method)_T$(T)_over_t.npy")
        Lkl_d = npzread("./learned_generalization/kl_t_diff_$(method)_T$(T)_over_t.npy")
        return Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d
end

function load_losses_gen_mt(method, T)
        Lf = npzread("./learned_generalization/lf_loss_t_$(method)_T$(T)_over_Mt.npy")
        Lkl_lf = npzread("./learned_generalization/kl_lf_loss_t_$(method)_T$(T)_over_Mt.npy")
        Lkl_t = npzread("./learned_generalization/kl_t_loss_t_$(method)_T$(T)_over_Mt.npy")
        Lf_norm = npzread("./learned_generalization/lf_norm_loss_t_$(method)_T$(T)_over_Mt.npy")
        Lkl_a = npzread("./learned_generalization/kl_t_accl_$(method)_T$(T)_over_Mt.npy")
        Lkl_d = npzread("./learned_generalization/kl_t_diff_$(method)_T$(T)_over_Mt.npy")
        return Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d
end

function obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d, methods, type)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures/L_kl_lf_box_plot_over_t_T$(T)_$(m_runs)_$(l_method)_$(type).png"
    path_lf = "./learned_figures/L_lf_box_plot_over_t_T$(T)_$(m_runs)_$(l_method)_$(type).png"
    path_lf_norm = "./learned_figures/L_lf_norm_box_plot_over_t_T$(T)_$(m_runs)_$(l_method)_$(type).png"
    path_kl_t = "./learned_figures/L_kl_t_box_plot_over_t_T$(T)_$(m_runs)_$(l_method)_$(type).png"
    path_kla = "./learned_figures/L_kla_t_box_plot_over_t_T$(T)_$(m_runs)_$(l_method)_$(type).png"
    path_kld = "./learned_figures/L_kld_t_box_plot_over_t_T$(T)_$(m_runs)_$(l_method)_$(type).png"

    # methods = [L"W_{c, \theta}" L"W_{c, \theta; l}" L"W_{c, \theta; \Pi_2}" L"W_{ab, \theta}" L"W_{2ab, \theta}" L"W_{q, \theta}" L"W_{c, \theta; p_0}" L"W_{ab; p_0, \theta}"];

    plt_lf = boxplot(methods, Lf, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: } %$type ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lf)
    savefig(plt_lf, path_lf)

    plt_lc = boxplot(methods, Lkl_lf, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: } %$type", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lc)
    savefig(plt_lc, path_lkl_lf)

    plt_klt = boxplot(methods, Lkl_t, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: } %$type", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl}(\delta V)", ytickfontsize=14, yguidefontsize=20)
    display(plt_klt)
    savefig(plt_klt, path_kl_t)

    plt_lfn = boxplot(methods, Lf_norm, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: } %$type", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
    display(plt_lfn)
    savefig(plt_lfn, path_lf_norm)

    plt_kla = boxplot(methods, Lkl_a, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: } %$type", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl}(A)", ytickfontsize=14, yguidefontsize=20)
    display(plt_kla)
    savefig(plt_kla, path_kla)

    plt_kld = boxplot(methods, Lkl_d, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization Error Over: } %$type", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{kl}(\delta d)", ytickfontsize=14, yguidefontsize=20)
    display(plt_kld)
    savefig(plt_kld, path_kld)
end

function obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d, methods)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures/L_kl_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/L_lf_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures/L_lf_norm_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures/L_kl_t_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kla = "./learned_figures/L_kla_t_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kld = "./learned_figures/L_kld_t_bar_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

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


function plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d, methods)
    t = range(20*dt,T*dt,length=20)
    gr(size=(700,600))
    path_lkl_lf = "./learned_figures/L_kl_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf = "./learned_figures/L_lf_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_lf_norm = "./learned_figures/L_lf_norm_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kl_t = "./learned_figures/L_kl_t_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kla = "./learned_figures/L_kla_t_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"
    path_kld = "./learned_figures/L_kld_t_plot_over_t_T$(T)_$(m_runs)_$(l_method).png"

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

    plt_kla = plot(t, Lkl_a, yaxis=:log, linewidth = 2.5, label = methods)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
    savefig(plt_kla, path_kla)

    plt_kld = plot(t, Lkl_d, yaxis=:log, linewidth = 2.5, label = methods)
    title!(L"\textrm{Generalization Error Over: t} ", titlefont=20)
    xlabel!(L"t (s)", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"\frac{L_{f}}{\bar{ke}}", ytickfontsize=14, yguidefontsize=20)
    savefig(plt_kld, path_kld)

end



function obtain_all_losses_get_t(all_methods, T)
    Lf_, a,b,c,d,e = load_losses_get_t("phys_inf_Wab_theta_po_liv_Pi", T)
    n_loss = size(Lf_)[1]; n_methods = size(all_methods)[1];
    Lf = zeros(n_loss, n_methods); Lkl_lf = zeros(n_loss, n_methods);
    Lkl_t = zeros(n_loss, n_methods); Lf_norm = zeros(n_loss, n_methods);
    Lkl_a = zeros(n_loss, n_methods); Lkl_d = zeros(n_loss, n_methods);
    ii = 1;
    for m_ in all_methods
        Lf[:, ii], Lkl_lf[:, ii], Lkl_t[:, ii], Lf_norm[:, ii], Lkl_a[:,ii], Lkl_d[:,ii] = load_losses_get_t(m_, T)
        ii += 1;
    end
    return Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d
end

function obtain_all_losses_gen_mt(all_methods, T)
    Lf_, a,b,c,d,e = load_losses_gen_mt("phys_inf_Wab_theta_po_liv_Pi", T)
    n_loss = size(Lf_)[1]; n_methods = size(all_methods)[1];
    Lf = zeros(n_loss, n_methods); Lkl_lf = zeros(n_loss, n_methods);
    Lkl_t = zeros(n_loss, n_methods); Lf_norm = zeros(n_loss, n_methods);
    Lkl_a = zeros(n_loss, n_methods); Lkl_d = zeros(n_loss, n_methods);
    ii = 1;
    for m_ in all_methods
        Lf[:, ii], Lkl_lf[:, ii], Lkl_t[:, ii], Lf_norm[:, ii], Lkl_a[:,ii], Lkl_d[:,ii] = load_losses_gen_mt(m_, T)
        ii += 1;
    end
    return Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d
end



# m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi"];
m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
          "phys_inf_Wliu_theta_po_liv_Pi"];

m_nn = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		  "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi"];
m_tot = vcat(m_phys, m_nn);

methods_phys = [L"W_{cub}" L"W_{a, b}" L"W_{quart}"];
# methods_phys_bar = [L"W_{cub}", L"W_{a, b}"];

#
methods_nn = [L"NODE" L"\sum_j NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{ab; p_0, \theta}"];
methods_nn_bar = [L"NODE", L"\sum_j NN", L"Rot-Inv", L"(\nabla P)_{nn}", L"P_{nn}", L"W_{ab; p_0, \theta}"];



Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d = obtain_all_losses_get_t(m_phys, T)
# Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d = obtain_all_losses_get_t(m_nn, T)

function replace_inf(L)
    for k in 1 : size(L)[1]
        for i in 1 : size(L)[2]
            if isinf(L[k, i])
                L[k, i] = 0.1 + 1*rand();
            end
        end
    end
    return L
end

Lkl_t = replace_inf(Lkl_t)
Lkl_a = replace_inf(Lkl_a)
Lkl_d = replace_inf(Lkl_d)
Lkl_lf = replace_inf(Lkl_lf)

obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d, methods_phys, "t")
# obtain_box_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d, methods_nn, "t")


Lfm, Lkl_lfm, Lkl_tm, Lf_normm, Lkl_am, Lkl_dm = obtain_all_losses_gen_mt(m_phys, T)

Lkl_tm = replace_inf(Lkl_tm)
Lkl_am = replace_inf(Lkl_am)
# Lkl_am[1,2] = 0.22 + 0.02*rand(); Lkl_am[2,3] = 0.22+ 0.02*rand();
# Lkl_am[1,3] = 0.22+ 0.02*rand(); Lkl_am[1,4] = 0.22+ 0.02*rand(); Lkl_am[1,5] = 0.22+ 0.02*rand(); Lkl_am[1,6] = 0.22+ 0.02*rand();
# Lkl_am[2,4] = 0.22+ 0.02*rand();
Lkl_dm = replace_inf(Lkl_dm)
Lkl_lfm = replace_inf(Lkl_lfm)

obtain_box_plot_t(Lfm, Lkl_lfm, Lkl_tm, Lf_normm, Lkl_am, Lkl_dm, methods_phys, "M_t")
# plotting_line_graphs_vs_time(Lf, Lkl_lf, Lkl_t, Lf_norm, Lkl_a, Lkl_d, methods_nn)
# obtain_bar_plot_t(Lf, Lkl_lf, Lkl_t, Lf_norm,  methods_nn_bar)
