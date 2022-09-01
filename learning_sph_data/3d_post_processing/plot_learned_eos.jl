c1 = "blue2"; c2 = "darkorange1"; c3 = "red3"; c4 = "purple3"; c5 = "springgreen2";
c6 = "sienna3"; c7 = "deepskyblue3"; c8 = "deeppink1"; c9 = "darkgreen";
cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9];

using Plots, LaTeXStrings, NPZ
using BSON: @load
using Flux

gr(size=(900,750));

#font sizes
legend_fs = 18; yaxis_fs = xaxis_fs = 20;
title_fs = 22; tick_fs = 18;

#linewidth and markersize
lw = 6.0; ms_ = 7.0;

function Pres(rho, c, g)
  return c^2 * (rho^g - 1.) / g ;
end

	height = 10;
	nn_data_dir = "./learned_models/output_data_forward_eos_nn_lf_Vrandn_itr1200_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height10_klswitch0"
	c_gt = c = 10; g = 7.0
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]



function compare_eos(p_h)
	path = "./learned_figures/learned_eos.png"
    Pnn_comp(ρ) = re(p_h)([ρ])[1]
	# max_rho = maximum(rhos_gt[1:11,:]);
	# min_rho = minimum(rhos_gt[1:11,:]);
	# mean_rho = mean(rhos_gt[1:11,:]);
	mean_rho = 1.035;
	# std_3 = 3*std(rhos_gt[1:11,:]);
	std_3 = 0.06;
	rho_std_p = mean_rho + std_3;
	rho_std_m = mean_rho - std_3;
	r_s = 0.90;
	r_e = 1.15;

	plt = plot(x -> Pres(x, c_gt, g), r_s, r_e, label=L"P_{truth}", color="black", linewidth = lw)
    plot!(x -> Pnn_comp(x), r_s, r_e, marker=:x, markersize=ms_, color=cs[5], markercolor = :black,
                  label=L"P_{nn(\theta)}(\rho)", linestyle=:dash, linewidth = lw,  legendfontsize=legend_fs)

	# vline!([min_rho, max_rho], linewidth = lw, color="orangered4", linestyle=:dash, label=L"(\rho_{min}, \rho_{max})")
	vline!([rho_std_p, rho_std_m], linewidth = lw, color="black", label=L"(\mu_{\rho} - 3\sigma_{\rho}, \mu_{\rho} + 3\sigma_{\rho})", linestyle=:dot)
	vline!([mean_rho], linewidth = lw, color="purple", linestyle=:dashdot, label=L"\rho_{avg}",  legendfontsize=legend_fs, legend=:topleft)

	title!(L"\textrm{Learned EoS with } L_f", titlefont=title_fs)
	xlabel!(L"\rho", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
	ylabel!(L"P(\rho)", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

   display(plt)
   savefig(plt, path)
end

compare_eos(p_fin)

function plot_loss()
	path = "./learned_figures/learned_eos_loss.png"
	loss_data = npzread("$(nn_data_dir)/loss.npy")
	plt = plot(loss_data, yaxis=:log, color="blue", label=L"L_f", linewidth = 3, legendfontsize=legend_fs)
	title!(L"\textrm{Losses over iterations }", titlefont=title_fs)
	xlabel!(L"\textrm{Iteration}", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
	ylabel!(L"L_f", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)
	savefig(plt, path)
	display(plt)
end
plot_loss()
