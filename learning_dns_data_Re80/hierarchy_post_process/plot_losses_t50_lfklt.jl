using Plots, NPZ, LaTeXStrings


l_method = "lf"
# m_all_l = [L"W_2(a, b)", L"W(a, b)", L"W_{cub}", L"W_{quart}", L"NODE", L"\sum NN", L"Rot-Inv", L"P_{nn}(\rho)", L"\nabla P_{nn}"];
m_all_l = [L"NODE", L"\sum NN", L"Rot-Inv", L"\nabla P_{nn}", L"P_{nn}(\rho)", L"W_{cub}", L"W_{quart}", L"W(a, b)", L"W_2(a, b)"];


function load_phys_dirs_names_t50()
	dir_t20 = "learned_models_t50_lfklt"
	d1 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_W2ab_theta_po_liv_Pi_lf_skl1_itr800_lr0.02_T50_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_W2ab_theta_po_liv_Pi_llf_klswitch1"
	d2 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wab_theta_po_liv_Pi_lf_skl1_itr800_lr0.02_T50_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wab_theta_po_liv_Pi_llf_klswitch1"
	d3 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wliu_theta_po_liv_Pi_lf_skl1_itr800_lr0.02_T50_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wliu_theta_po_liv_Pi_llf_klswitch1"
	d4 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_theta_po_liv_Pi_lf_skl1_itr800_lr0.02_T50_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mphys_inf_theta_po_liv_Pi_llf_klswitch1"
	return d1, d2, d3, d4
end

d1, d2, d3, d4= load_phys_dirs_names_t50()

function load_loss(dir_name)
	loss = npzread("$(dir_name)/loss.npy")
	return loss
end

loss_w2ab = load_loss(d1);
loss_wab = load_loss(d2);
loss_wliu = load_loss(d3);
loss_wcub = load_loss(d4);

include("./color_scheme_utils.jl")
include("./plot_dims.jl")


function plot_losses()
	xs = 5 : 5 : 800

	cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
	id_s = 4; idx = vcat(1:15, 16:id_s:160)
	# plt = plot(xs[idx], loss_node[idx], label=m_all_l[1], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[1], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_nnsm[idx], label=m_all_l[2], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[2], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_rot[idx], label=m_all_l[3], xaxis =:log, yaxis =:log,
	# 	  markershapes =:star5, ms = ms_, linewidth = lw, color=cs[3], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_grap[idx], label=m_all_l[4], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[4], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_eos[idx], label=m_all_l[5], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[5], legendfontsize=12)
	#

	plt = plot(xs[idx], loss_wcub[idx], label=m_all_l[6], xaxis =:log,  yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[6], legendfontsize=14)
	#
	plot!(xs[idx], loss_wliu[idx], label=m_all_l[7], xaxis =:log,  yaxis =:log,
		  markershapes =:diamond, ms = ms_, linewidth = lw, color=cs[7], legendfontsize=14)
    #
	plot!(xs[idx], loss_wab[idx], label=m_all_l[8], xaxis =:log,  yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[8], legendfontsize=12)
	#
	plot!(xs[idx], loss_w2ab[idx], label=m_all_l[9], xaxis =:log,  yaxis =:log,
			markershapes =:rect, ms = ms_, linewidth = lw, color=cs[9], legendfontsize=legend_fs)
	#
	title!(L"\textrm{Losses ~ over ~ Iterations ~ } L_f \rightarrow L_{kl}", titlefont=title_fs)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Iteration}", xtickfontsize=tick_fs, xguidefontsize=xaxis_fs)
    ylabel!(L"\textrm{Field ~ Loss: } L_f", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

    display(plt)
    out_path = "./learned_figures_t50_lfklt/losses_t50_over_itrs_$(l_method).png"
    savefig(plt, out_path)
end



plot_losses()



function plot_losses_tail(idx_s, idx_e)
	xs = 5 : 5 : 800;
	cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
	idx = idx_s : idx_e
	# plt = plot(xs[idx], loss_node[idx], label=m_all_l[1], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[1], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_nnsm[idx], label=m_all_l[2], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[2], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_rot[idx], label=m_all_l[3], xaxis =:log, yaxis =:log,
	# 	  markershapes =:star5, ms = ms_, linewidth = lw, color=cs[3], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_grap[idx], label=m_all_l[4], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[4], legendfontsize=12)
	# #
	# plot!(xs[idx], loss_eos[idx], label=m_all_l[5], xaxis =:log, yaxis =:log,
	# 	  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[5], legendfontsize=12)
	#

	plt = plot(xs[idx], loss_wcub[idx], label=m_all_l[6],  yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[6], legendfontsize=14)
	#
	plot!(xs[idx], loss_wliu[idx], label=m_all_l[7], yaxis =:log,
		  markershapes =:diamond, ms = ms_, linewidth = lw, color=cs[7], legendfontsize=14)
    #
	plot!(xs[idx], loss_wab[idx], label=m_all_l[8], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[8], legendfontsize=12)
	#
	plot!(xs[idx], loss_w2ab[idx], label=m_all_l[9], yaxis =:log,
			markershapes =:rect, ms = ms_, linewidth = lw, color=cs[9], legendfontsize=legend_fs)
	#
	title!(L"\textrm{Losses ~ over ~ Iterations ~ } L_f \rightarrow L_{kl}", titlefont=title_fs)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Iteration}", xtickfontsize=tick_fs, xguidefontsize=xaxis_fs)
    ylabel!(L"\textrm{Field ~ Loss: } L_f", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

    display(plt)
    out_path = "./learned_figures_t50_lfklt/losses_t50_tail_over_itrs_$(l_method).png"
    savefig(plt, out_path)
end

plot_losses_tail(60, 160)
