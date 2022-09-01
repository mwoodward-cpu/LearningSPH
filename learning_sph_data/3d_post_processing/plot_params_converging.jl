using Plots, LaTeXStrings, NPZ

gr(size=(900,750));

α = 1.0
β = 2.0*α  #usual value of params for alpha and β but these depend on problem
θ = 5e-1;
c = 10.0
g = 7.0
h = 0.335
cdt = 0.4;

#font sizes
legend_fs = 18; yaxis_fs = xaxis_fs = 20;
title_fs = 22; tick_fs = 18;

#linewidth and markersize
lw = 6.0; ms_ = 7.0;

path_mod = "./learned_models/output_data_forward_phys_inf_lf_Vrandn_itr3000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch1/"
α_out = npzread("$(path_mod)/alpha_out.npy")
β_out = npzread("$(path_mod)/beta_out.npy")
c_out = npzread("$(path_mod)/c_out.npy")
g_out = npzread("$(path_mod)/g_out.npy")


vis_rate = 1;
function plot_4g_param()
	path = "learned_figures/validation.png"
    println("*************** generating plots ******************")

    c_gt_data = c_gt * ones(size(c_out))
    α_gt_data = α * ones(size(α_out))
    β_gt_data = β * ones(size(β_out))
    g_gt_data = g * ones(size(g_out))
    xs = 1 : vis_rate : (vis_rate * size(c_out)[1])

    plt = plot(xs, c_out, label=L"\hat{c}", color="green", linewidth = lw)
    plot!(xs, c_gt_data, label=L"c", color="green", linestyle=:dash, linewidth = lw)

    plot!(xs, α_out, label=L"\hat{\alpha}", color="blue", linewidth = lw)
    plot!(xs, α_gt_data, label=L"\alpha", linestyle=:dash, color = "blue", linewidth = lw)

    plot!(xs, β_out, label=L"\hat{\beta}", color="purple", linewidth = lw)
    plot!(xs, β_gt_data, label=L"\beta", linestyle=:dash, color = "purple", linewidth = lw)

    plot!(xs, g_out, label=L"\hat{\gamma}", color="gold4", linewidth = lw)
    plot!(xs, g_gt_data, label=L"\gamma", linestyle=:dash, color = "gold4", linewidth = lw, legendfontsize=legend_fs)

    title!(L"\textrm{Validation of Learning on SPH data}", titlefont=title_fs)
	xlabel!(L"\textrm{Iterations}", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
	ylabel!(L"\textrm{Parameters}", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

    display(plt)
    savefig(plt, path)
end

plot_4g_param()
