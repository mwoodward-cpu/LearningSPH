using Plots, NPZ, LaTeXStrings


l_method = "lf"
# m_all_l = [L"W_2(a, b)", L"W(a, b)", L"W_{cub}", L"W_{quart}", L"NODE", L"\sum NN", L"Rot-Inv", L"P_{nn}(\rho)", L"\nabla P_{nn}"];
m_all_l = [L"NODE", L"\sum NN", L"Rot-Inv", L"\nabla P_{nn}", L"P_{nn}(\rho)", L"W_{cub}", L"W_{quart}", L"W(a, b)", L"W_2(a, b)"];


function load_dirs_names_t20()
	dir_t20 = "learned_models_t20_minl"
	d1 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_W2ab_theta_po_liv_Pi_lf_skl0_itr2300_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_W2ab_theta_po_liv_Pi_llf_klswitch0"
	d2 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wab_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wab_theta_po_liv_Pi_llf_klswitch0"
	d3 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wliu_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wliu_theta_po_liv_Pi_llf_klswitch0"
	d4 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mphys_inf_theta_po_liv_Pi_llf_klswitch0"
	d5 = "$(dir_t20)/output_data_unif_tracers_forward_node_norm_theta_liv_Pi_lf_itr800_lr0.002_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height8_mnode_norm_theta_liv_Pi_llf_klswitch0"
	d6 = "$(dir_t20)/output_data_unif_tracers_forward_nnsum2_norm_theta_liv_Pi_lf_itr600_lr0.01_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mnnsum2_norm_theta_liv_Pi_llf_klswitch0"
	d7 = "$(dir_t20)/output_data_unif_tracers_forward_rot_inv_theta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mrot_inv_theta_liv_Pi_llf_klswitch0"
	d8 = "$(dir_t20)/output_data_unif_tracers_forward_eos_nn_theta_alpha_beta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height8_meos_nn_theta_alpha_beta_liv_Pi_llf_klswitch0"
	d9 = "$(dir_t20)/output_data_unif_tracers_forward_grad_p_theta_alpha_beta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mgrad_p_theta_alpha_beta_liv_Pi_llf_klswitch0"
	return d1, d2, d3, d4, d5, d6, d7, d8, d9
end

d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()
all_dirs = [d1, d2, d3, d4, d5, d6, d7, d8, d9];

function load_loss(dir_name)
	loss = npzread("$(dir_name)/loss.npy")
	return loss
end

loss_w2ab = load_loss(d1)[1:180];
loss_wab = load_loss(d2)[1:180];
loss_wliu = load_loss(d3)[1:180];
loss_wcub = load_loss(d4)[1:180];
loss_node = load_loss(d5);
loss_nnsm = load_loss(d6);
loss_rot = load_loss(d7)[1:180];
loss_eos = load_loss(d8)[1:180];
loss_grap = load_loss(d9);

loss_nnsm = vcat(loss_nnsm, loss_nnsm[end]*ones(60));
loss_grap = vcat(loss_grap, loss_grap[end]*ones(150));
loss_node = vcat(loss_node, loss_node[end]*ones(20))

include("./color_scheme_utils.jl")

function plot_losses()
    # gr(size=(900,750))
	xs = 5 : 5 : 900

	#palette=:darktest; palette =:balance
	cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
	# palette(colors, 9);
	# lw = 5.8; ms_ = 6.8;
	id_s = 4; idx = vcat(1:15, 16:id_s:180)
	plt = plot(xs[idx], loss_node[idx], label=m_all_l[1], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[1], legendfontsize=12)
	#
	plot!(xs[idx], loss_nnsm[idx], label=m_all_l[2], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[2], legendfontsize=12)
	#
	plot!(xs[idx], loss_rot[idx], label=m_all_l[3], xaxis =:log, yaxis =:log,
		  markershapes =:star5, ms = ms_, linewidth = lw, color=cs[3], legendfontsize=12)
	#
	plot!(xs[idx], loss_grap[idx], label=m_all_l[4], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[4], legendfontsize=12)
	#
	plot!(xs[idx], loss_eos[idx], label=m_all_l[5], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[5], legendfontsize=12)
	#
	plot!(xs[idx], loss_wcub[idx], label=m_all_l[6], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[6], legendfontsize=14)
	#
	plot!(xs[idx], loss_wliu[idx], label=m_all_l[7], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[7], legendfontsize=14)
    #
	plot!(xs[idx], loss_wab[idx], label=m_all_l[8], xaxis =:log, yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[8], legendfontsize=12)
	#
	plot!(xs[idx], loss_w2ab[idx], label=m_all_l[9], xaxis =:log, yaxis =:log,
			markershapes =:rect, ms = ms_, linewidth = lw, color=cs[9], legendfontsize=legend_fs)
	#
	title!(L"\textrm{Losses ~ over ~ Iterations}", titlefont=title_fs)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Iteration}", xtickfontsize=tick_fs, xguidefontsize=xaxis_fs)
    ylabel!(L"\textrm{Field ~ Loss: } L_f", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

    display(plt)
    out_path = "./learned_figures/losses_t20_over_itrs_$(l_method).png"
    savefig(plt, out_path)
end
plot_losses()



function plot_losses_tail(idx_s, idx_e)
	xs = 5 : 5 : 900
	cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
	idx = idx_s : 4 : idx_e;

	plt = plot(xs[idx], loss_node[idx], label=m_all_l[1], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[1], legendfontsize=12)
	#
	plot!(xs[idx], loss_nnsm[idx], label=m_all_l[2], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[2], legendfontsize=12)
	#
	plot!(xs[idx], loss_rot[idx], label=m_all_l[3], yaxis =:log,
		  markershapes =:star5, ms = ms_, linewidth = lw, color=cs[3], legendfontsize=12)
	#
	plot!(xs[idx], loss_grap[idx], label=m_all_l[4], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[4], legendfontsize=12)
	#
	plot!(xs[idx], loss_eos[idx], label=m_all_l[5], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[5], legendfontsize=12)
	#
	plot!(xs[idx], loss_wcub[idx], label=m_all_l[6], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[6], legendfontsize=14)
	#
	plot!(xs[idx], loss_wliu[idx], label=m_all_l[7], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[7], legendfontsize=14)
    #
	plot!(xs[idx], loss_wab[idx], label=m_all_l[8], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, color=cs[8], legendfontsize=12)
	#
	plot!(xs[idx], loss_w2ab[idx], label=m_all_l[9], yaxis =:log, ylims=(6.918e-5, 1.9e-4),
			markershapes =:rect, ms = ms_, linewidth = lw, color=cs[9], legendfontsize=legend_fs)
	#
	title!(L"\textrm{Losses ~ over ~ Iterations}", titlefont=title_fs)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Iteration}", xtickfontsize=tick_fs, xguidefontsize=xaxis_fs)
    ylabel!(L"\textrm{Field ~ Loss: } L_f", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

    display(plt)
    out_path = "./learned_figures/losses_t20_tail_over_itrs_$(l_method).png"
    savefig(plt, out_path)
end
plot_losses_tail(100, 180)





function plot_losses_phys()
    gr(size=(700,600))
	xs = 5 : 5 : 800

	lw = 2.3; ms_ = 5.0;
    plt = plot(xs, loss_w2ab, label=m_all_l[1], yaxis =:log, markershapes =:auto, ms = ms_, linewidth = lw)
	plot!(xs, loss_wab, label=m_all_l[2], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw)
	#
	plot!(xs, loss_wliu, label=m_all_l[3], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, legendfontsize=12)
    #
	plot!(xs, loss_wcub, label=m_all_l[4], yaxis =:log,
		  markershapes =:auto, ms = ms_, linewidth = lw, legendfontsize=12)
	#
	title!(L"\textrm{Comparing Loss over Iterations}", titlefont=22)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Iteration}", xtickfontsize=16, xguidefontsize=20)
    ylabel!(L"L_f", ytickfontsize=16, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/t20_losses_over_itrs_$(l_method)_phys.png"
    savefig(plt, out_path)
end

# plot_losses_phys()




#using set palette, but loose control of setting colors on subset.
# function plot_losses()
#     gr(size=(600,500))
# 	xs = 5 : 5 : 800
#
# 	#palette=:darktest; palette =:balance
# 	lw = 4.5; ms_ = 7.0; id_s = 4; idx = vcat(1:15, 16:id_s:160)
# 	plt = plot(xs[idx], loss_node[idx], label=m_all_l[5], xaxis =:log, yaxis =:log,
# 		  markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
# 	plot!(xs[idx], loss_nnsm[idx], label=m_all_l[6], xaxis =:log, yaxis =:log,
# 		  markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
# 	plot!(xs[idx], loss_rot[idx], label=m_all_l[7], xaxis =:log, yaxis =:log,
# 		  markershapes =:rect, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
# 	plot!(xs[idx], loss_grap[idx], label=m_all_l[9], xaxis =:log, yaxis =:log,
# 		  markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
# 	plot!(xs[idx], loss_eos[idx], label=m_all_l[8], xaxis =:log, yaxis =:log,
# 		  markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
#     plot!(xs[idx], loss_w2ab[idx], label=m_all_l[1], xaxis =:log, yaxis =:log,
# 			markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
# 	plot!(xs[1:id_s:end], loss_wab[1:id_s:end], label=m_all_l[2], xaxis =:log, yaxis =:log,
# 		  markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
# 	#
# 	plot!(xs[idx], loss_wcub[idx], label=m_all_l[3], xaxis =:log, yaxis =:log,
# 		  markershapes =:xcross, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=14)
# 	#
# 	plot!(xs[idx], loss_wliu[idx], label=m_all_l[4], xaxis =:log, yaxis =:log,
# 		  markershapes =:auto, ms = ms_, linewidth = lw, palette =:tab10, legendfontsize=12)
#     #
# 	title!(L"\textrm{Comparing ~ Loss ~ over ~ Iterations}", titlefont=18)
#     # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
#     xlabel!(L"\textrm{Iteration}", xtickfontsize=12, xguidefontsize=14)
#     ylabel!(L"\textrm{Field loss: } L_f", ytickfontsize=12, yguidefontsize=14)
#
#     display(plt)
#     out_path = "./learned_figures/losses_t20_over_itrs_$(l_method).png"
#     savefig(plt, out_path)
# end

# plot_losses()
