

using Plots, NPZ, LaTeXStrings
using StatsPlots
ENV["GKSwstype"]="100" #set env variable for UAHPC


#-----Load data

function load_losses(method)
	L = npzread("./learned_generalization/loss_t_$(method)_losskl_lf_lf.npy")
	Lg = npzread("./learned_generalization/loss_g_t_$(method)_losskl_lf_lf.npy")
	return L, Lg
end

function load_losses_theta(method)
	L = npzread("./learned_generalization/Lθ_$(method)_kl_lf_lf.npy")
	Lg = npzread("./learned_generalization/Lgθ_$(method)_kl_lf_lf.npy")
	return L, Lg
end

#---Plotting

function plot_rot_error(θ_idx)
    gr(size=(800,800))
    println("*************** plot gen_error ******************")

    rot_errs = [rot_rfθ_node[θ_idx], rot_rfθ_nnsum[θ_idx], rot_rfθ_nnsum2[θ_idx],
                rot_rfθ_gradp[θ_idx], rot_rfθ_rotinv[θ_idx], rot_rfθ_eos[θ_idx], rot_rfθ_phy[θ_idx]];

    methods = [L"NODE", L"\sum_j NN_{ij}", L"\sum_j NN2_{ij}", L"\nabla P_{NN}", L"RI_{NN}", L"EoS_{NN}", L"Phys"];
    # methods = [L"NODE" L"\sum_i NN_{ij}" L"\sum_i NN2_{ij}" L"\nabla P_{nn}" L"rotNN_{inv}" L"EoS_{nn}" L"Phys"];

    plt_rot = bar(methods, rot_errs, yaxis=:log, color="blue", label=false)
    title!(L"\textrm{Rotational - Invariance - Errors }", titlefont=16)
    xlabel!(L"\textrm{Method}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||F(RX) - RF(X)||_2", ytickfontsize=10, yguidefontsize=14)
    display(plt_rot)
    path_rot = "./gen_error_figures/bar_plot_rot_err_coarse$(coarse).png"
    savefig(plt_rot, path_rot)
end



function plot_gen_error(θ_idx, t_idx)
    gr(size=(800,800))
    println("*************** plot gen_error ******************")

    gen_errsθ = [Lgθ_node[θ_idx], Lgθ_nnsum[θ_idx], Lgθ_nnsum2[θ_idx],
                Lgθ_gradp[θ_idx], Lgθ_rotinv[θ_idx], Lgθ_eos[θ_idx], Lgθ_phy[θ_idx]];
    lossesθ = [Lθ_node[θ_idx], Lθ_nnsum[θ_idx], Lθ_nnsum2[θ_idx],
                Lθ_gradp[θ_idx], Lθ_rotinv[θ_idx], Lθ_eos[θ_idx], Lθ_phy[θ_idx]];
    # rot_errs = [rot_rfθ_node[θ_idx], rot_rfθ_nnsum[θ_idx], rot_rfθ_nnsum2[θ_idx],
    #             rot_rfθ_gradp[θ_idx], rot_rfθ_rotinv[θ_idx], rot_rfθ_eos[θ_idx], rot_rfθ_phy];

    gen_errs_t = [Lgt_node[t_idx], Lgt_nnsum2[t_idx], Lgt_nnsum[t_idx],
                Lgt_gradp[t_idx], Lgt_rotinv[t_idx], Lgt_eos[t_idx], Lgt_phy[t_idx]];
    losses_t = [Lt_node[t_idx], Lt_nnsum[t_idx], Lt_nnsum2[t_idx],
                Lt_gradp[t_idx], Lt_rotinv[t_idx], Lt_eos[t_idx],  Lt_phy[t_idx]];

    methods = [L"NODE", L"\sum_j NN_{ij}", L"\sum_j NN2_{ij}", L"\nabla P_{NN}", L"RI_{NN}", L"EoS_{NN}", L"Phys"];

    plt_θ = bar(methods, gen_errsθ, yaxis=:log, color="blue", label=false)
    title!(L"\textrm{Comparing Generalization Error Over } \theta", titlefont=16)
    xlabel!(L"\textrm{Method}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{KL}", ytickfontsize=10, yguidefontsize=14)
    display(plt_θ)

    plt_t = bar(methods, gen_errs_t, yaxis=:log, color="blue", label=false)
    title!(L"\textrm{Comparing Generalization Error Over } t", titlefont=16)
    xlabel!(L"\textrm{Method}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{KL}", ytickfontsize=10, yguidefontsize=14)
    display(plt_t)

    path_θ = "./gen_error_figures/bar_plot_gen_errθ_coarse$(coarse).png"
    path_t = "./gen_error_figures/bar_plot_gen_errt_coarse$(coarse).png"
    savefig(plt_θ, path_θ)
    savefig(plt_t, path_t)
end




function obtain_box_plot_t(Lgs)
    gr(size=(700,700))
    # path_lg = "./learned_figures/L_kl_lf_box_plot_over_t.png"
	path_lg = "./learned_figures/L_lf_box_plot_over_t.png"

    methods = [L"NODE" L"\sum_j NN_{ij}" L"RI_{NN}" L"\nabla P_{nn}" L"EoS_{NN}" L"Phys"];

    plt = boxplot(methods, Lgs, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization-Error-Over: t} ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    # ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
	ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    savefig(plt, path_lg)

end

function obtain_box_plot_theta(Lgs)
    gr(size=(700,700))
	# path_lg = "./learned_figures/L_kl_lf_box_plot_over_theta.png"
	path_lg = "./learned_figures/L_lf_box_plot_over_theta.png"

	methods = [L"NODE" L"\sum_j NN_{ij}" L"RI_{NN}" L"\nabla P_{nn}" L"EoS_{NN}" L"Phys"];

    plt = boxplot(methods, Lgs, yaxis=:log, legend=false, outliers=false)
    title!(L"\textrm{Generalization-Error-Over: } \theta ", titlefont=20)
    xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"L_{f}", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    savefig(plt, path_lg)

end






#---------Plotting

# plot_gen_error(3, 5)
# plot_rot_error(2)

L_node_t, Lg_node_t = load_losses("node")
L_nnsum_t, Lg_nnsum_t = load_losses("nnsum")
L_rot_inv_t, Lg_rot_inv_t = load_losses("rot_inv")
L_grad_p_t, Lg_grad_p_t = load_losses("grad_p")
L_eos_nn_t, Lg_eos_nn_t = load_losses("eos_nn")
L_phys_inf_t, Lg_phys_inf_t = load_losses("phys_inf")


Lθ_node_t, Lgθ_node_t = load_losses_theta("node")
Lθ_nnsum_t, Lgθ_nnsum_t = load_losses_theta("nnsum")
Lθ_rot_inv_t, Lgθ_rot_inv_t = load_losses_theta("rot_inv")
Lθ_grad_p_t, Lgθ_grad_p_t = load_losses_theta("grad_p")
Lθ_eos_nn_t, Lgθ_eos_nn_t = load_losses_theta("eos_nn")
Lθ_phys_inf_t, Lgθ_phys_inf_t = load_losses_theta("phys_inf")


L_kl_lf_gts = hcat(L_node_t, L_nnsum_t, L_rot_inv_t, L_grad_p_t,
            L_eos_nn_t, L_phys_inf_t);

L_kl_lf_gθs = hcat(Lθ_node_t, Lθ_nnsum_t, Lθ_rot_inv_t, Lθ_grad_p_t,
			Lθ_eos_nn_t, Lθ_phys_inf_t);

#
L_lf_gts = hcat(Lg_node_t, Lg_nnsum_t, Lg_rot_inv_t, Lg_grad_p_t,
            Lg_eos_nn_t, Lg_phys_inf_t);

L_lf_gθs = hcat(Lgθ_node_t, Lgθ_nnsum_t, Lgθ_rot_inv_t, Lgθ_grad_p_t,
			Lgθ_eos_nn_t, Lgθ_phys_inf_t);


# obtain_box_plot_t(L_kl_lf_gts)
# obtain_box_plot_theta(L_kl_lf_gθs)

# obtain_box_plot_t(L_lf_gts)
# obtain_box_plot_theta(L_lf_gθs)
#










#
#
#
# #------ Ploting functions
#
#
# function plotting_Lg()
#     gr(size=(500,500))
#     x_s = -width
#     x_e = width
#     G_pred(x) = kde(x, Vel_inc_pred[T, :, 1])
#     plt = plot(x->G_u(x), x_s, x_e, label=L"G(\tau, d)",
#                 color="indigo", linewidth = 2.5)
#     plot!(x->G_pred(x), x_s, x_e, marker=:x, markersize=4, color="forestgreen",
#           markercolor = :black, label=L"G_{\theta}(\tau, d)",
#           linestyle=:dash, linewidth = 2.5)
#
#     title!(L"\textrm{Comparing - } G(\tau,z) \textrm{ - and - } \hat{G}_{\theta}(\tau,z)", titlefont=18)
#     xlabel!(L"\textrm{Velocity - increment}", xtickfontsize=12, xguidefontsize=20)
#     ylabel!(L"G_u \textrm{ - distribution}", ytickfontsize=12, yguidefontsize=20)
#
#     display(plt)
#     if path == "train"
#         out_path = "./figures/$(method)_Gtrain_kde_$(file_title)_θ$(θ_in).png"
#     elseif path == "test"
#         out_path = "./figures/$(method)_Gtest_kde_$(file_title)_θ$(θ_in).png"
#     end
#     savefig(plt, out_path)
# end
#
#
# function plot_lg_comp()
#     gr(size=(600,500))
#     println("*************** plot gen_error ******************")
#     plt = bar(methods, gen_errs, yaxis=:log, color="blue", label=false)
#
#     title!(L"\textrm{Comparing - Generalization - Error}", titlefont=16)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=16)
#     ylabel!(L"\textrm{KL}", ytickfontsize=10, yguidefontsize=16)
#
#     display(plt)
#     savefig(plt, "./$(out_file_title)/figures/$(out_file_title)_gen_err.png")
# end

#
# #----symmetry errors
#
# function load_symmetry_errors(method)
# 	rot = npzread("./learned_generalization/rot_RF_$(method)_loss$(loss_method).npy")
# 	gal = npzread("./learned_generalization/gal_inv_$(method)_loss$(loss_method).npy")
# 	return rot, gal
# end
#
# rot_node, gal_node = load_symmetry_errors("node")
# rot_nnsum, gal_nnsum = load_symmetry_errors("nnsum")
# rot_rot_inv, gal_rot_inv = load_symmetry_errors("rot_inv")
# rot_grad_p, gal_grad_p = load_symmetry_errors("grad_p")
# rot_eos_nn, gal_eos_nn = load_symmetry_errors("eos_nn")
# rot_phys_inf, gal_phys_inf = load_symmetry_errors("phys_inf")
#
# rots = vcat(rot_node, rot_nnsum, rot_rot_inv, rot_grad_p,
#             rot_eos_nn, rot_phys_inf);
#
# gals = vcat(gal_node, gal_nnsum, gal_rot_inv, gal_grad_p,
# 			gal_eos_nn, gal_phys_inf);
#
#
# function plot_rot_gal()
#     gr(size=(700,700))
#     println("*************** plot gen_error ******************")
# 	methods = [L"NODE", L"\sum_j NN_{ij}", L"RI_{NN}", L"\nabla P_{NN}", L"EoS_{NN}", L"Phys"];
#
#     plt = bar(methods, rots, yaxis=:log, color="blue", label=false)
#
#     title!(L"\textrm{Rotational - Symmetry - Error}", titlefont=20)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"||F(RX) - RF(X)||_2", ytickfontsize=14, yguidefontsize=20)
#
# 	# title!(L"\textrm{Generalization-Error-Over: t} ", titlefont=20)
#     # xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     # ylabel!(L"L_{kl} + L_{f}", ytickfontsize=14, yguidefontsize=20)
#
# 	 display(plt)
# 	 savefig(plt, "./learned_figures/rots_bar_err.png")
#
#
# 	plt2 = bar(methods, gals, yaxis=:log, color="blue", label=false)
#
#     title!(L"\textrm{Translational - Symmetry - Error}", titlefont=20)
#     xlabel!(L"\textrm{Method}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"||F(X) - F(X - s)||_2", ytickfontsize=14, yguidefontsize=20)
#
# 	  display(plt2)
# 	  savefig(plt2, "./learned_figures/gals_bar_err.png")
# end
#
#
# plot_rot_gal()
#

# #
#
# #-----Plotting vel_incs
# traj_gt_path = "./learned_data/traj_N4096_T120_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
# vels_gt_path = "./learned_data/vels_N4096_T120_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
# rhos_gt_path = "./learned_data/rhos_N4096_T120_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
# traj_gt = npzread(traj_gt_path);
# vels_gt = npzread(vels_gt_path);
# rhos_gt = npzread(rhos_gt_path);
#
# N = size(traj_gt)[2];
# D = size(traj_gt)[3];
# m = (2. * pi)^D / N; #so that ρ₀ = 1;
#
# T = 120;
#
# include("./kde_G_3d.jl")
# loss_method = "kl_lf";
# function load_vel_inc(method)
# 	Vel_inc = npzread("./learned_generalization/vel_inc_pred_$(method)_loss$(loss_method)_t120.npy")
# 	return Vel_inc
# end
# function load_diff(method)
# 	diff = npzread("./learned_generalization/diff_pred_$(method)_loss$(loss_method)_t120.npy")
# 	return diff
# end
#
# vel_inc_node = load_vel_inc("node")
# vel_inc_nnsum = load_vel_inc("nnsum")
# vel_inc_rot  = load_vel_inc("rot_inv")
# vel_inc_grad = load_vel_inc("grad_p")
# vel_inc_eos  = load_vel_inc("eos_nn")
# vel_inc_phys = load_vel_inc("phys_inf")
#
# G_node(x) = kde(x, vec(vel_inc_node[T, :, :]))
# G_nnsum(x) = kde(x, vec(vel_inc_nnsum[T, :, :]))
# G_rot(x) = kde(x, vec(vel_inc_rot[T, :, :]))
# G_grad(x) = kde(x, vec(vel_inc_grad[T, :, :]))
# G_eos(x) = kde(x, vec(vel_inc_eos[T, :, :]))
# G_phys(x) = kde(x, vec(vel_inc_phys[T, :, :]))
#
# function comparing_GV_log(width=1.2)
#     gr(size=(700,700))
#     x_s = -width
#     x_e = width
# 	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
#     plt = plot(x->G_node(x), x_s, x_e, label=L"\hat{G}_{node}",
#           linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
# 	plot!(x->G_nnsum(x), x_s, x_e, label=L"\hat{G}_{nnsum}",
# 	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
# 	plot!(x->G_rot(x), x_s, x_e, label=L"\hat{G}_{rot}",
# 	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
#     plot!(x->G_grad(x), x_s, x_e, label=L"\hat{G}_{\nabla P}",
# 	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
# 	plot!(x->G_eos(x), x_s, x_e, label=L"\hat{G}_{EoS}",
# 	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
# 	plot!(x->G_phys(x), x_s, x_e, marker=:x, markersize=7,
# 	      markercolor = :black, label=L"\hat{G}_{phys}",
# 	      linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
# 	plot!(x->G_gt(x), x_s, x_e, label=L"G_{truth}",
# 		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")
#
# 	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=20)
# 	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
# 	xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=14, xguidefontsize=20)
# 	ylabel!(L"log(G_{\delta V})", ytickfontsize=14, yguidefontsize=20)
#
#     display(plt)
#     out_path = "./learned_figures/Gv_trained_comp_log_$(loss_method)_$(method)_kde_D$(D)_N$(N)_θ$(θ)_t$(T).png"
#
#     savefig(plt, out_path)
# end
#
#
# function comparing_GV(width=1.2)
#     gr(size=(700,700))
#     x_s = -width
#     x_e = width
# 	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
#     plt = plot(x->G_node(x), x_s, x_e, label=L"\hat{G}_{NODE}",
#           linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
# 	plot!(x->G_nnsum(x), x_s, x_e, label=L"\hat{G}_{NNsum}",
# 	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
# 	plot!(x->G_rot(x), x_s, x_e, label=L"\hat{G}_{RotInv}",
# 	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
#     plot!(x->G_grad(x), x_s, x_e, label=L"\hat{G}_{\nabla P}",
# 	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
# 	plot!(x->G_eos(x), x_s, x_e, label=L"\hat{G}_{EoS}",
# 	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
# 	plot!(x->G_phys(x), x_s, x_e, marker=:x, markersize=7,
# 	      markercolor = :black, label=L"\hat{G}_{PhysInf}",
# 	      linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
# 	plot!(x->G_gt(x), x_s, x_e, label=L"G_{truth}",
# 		  linewidth = 2.5, legendfontsize=10, color="indigo")
#
#
# 	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=20)
#     # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
#     xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"G_{\delta V}", ytickfontsize=14, yguidefontsize=20)
#
#     display(plt)
#     out_path = "./learned_figures/Gv_trained_comp_$(loss_method)_$(method)_kde_D$(D)_N$(N)_θ$(θ)_t$(T).png"
#
#     savefig(plt, out_path)
# end
#
# comparing_GV(2.4)
# comparing_GV_log(2.0)
#
#
# # T = 10;
#
# diff_node = load_diff("node")
# diff_nnsum = load_diff("nnsum")
# diff_rot  = load_diff("rot_inv")
# diff_grad = load_diff("grad_p")
# diff_eos  = load_diff("eos_nn")
# diff_phys = load_diff("phys_inf")
#
# Gd_node(x) = kde(x, vec(diff_node[T, :]))
# Gd_nnsum(x) = kde(x, vec(diff_nnsum[T, :]))
# Gd_rot(x) = kde(x, vec(diff_rot[T, :]))
# Gd_grad(x) = kde(x, vec(diff_grad[T, :]))
# Gd_eos(x) = kde(x, vec(diff_eos[T, :]))
# Gd_phys(x) = kde(x, vec(diff_phys[T, :]))
#
# function comparing_Gd_log(width=1.2)
#     gr(size=(700,700))
#     x_s = -0.001
#     x_e = width
# 	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
#     plt = plot(x->Gd_node(x), x_s, x_e, label=L"\hat{G}_{node}",
#           linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
# 	plot!(x->Gd_nnsum(x), x_s, x_e, label=L"\hat{G}_{nnsum}",
# 	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
# 	plot!(x->Gd_rot(x), x_s, x_e, label=L"\hat{G}_{rot}",
# 	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
#     plot!(x->Gd_grad(x), x_s, x_e, label=L"\hat{G}_{\nabla P}",
# 	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
# 	plot!(x->Gd_eos(x), x_s, x_e, label=L"\hat{G}_{EoS}",
# 	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
# 	plot!(x->Gd_phys(x), x_s, x_e, marker=:x, markersize=7,
# 	      markercolor = :black, label=L"\hat{G}_{phys}",
# 	      linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
# 	plot!(x->Gd_gt(x), x_s, x_e, label=L"G_{truth}",
# 		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")
#
# 	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=20)
# 	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
# 	xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=14, xguidefontsize=20)
# 	ylabel!(L"log(G)", ytickfontsize=14, yguidefontsize=20)
#
#     display(plt)
#     out_path = "./learned_figures/Gd_trained_comp_log_$(loss_method)_$(method)_kde_D$(D)_N$(N)_θ$(θ)_t$(T).png"
#
#     savefig(plt, out_path)
# end
#
#
# function comparing_Gd(width=1.2)
#     gr(size=(700,700))
#     x_s = -0.01
#     x_e = width
# 	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
#     plt = plot(x->Gd_node(x), x_s, x_e, label=L"\hat{G}_{NODE}",
#           linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
# 	plot!(x->Gd_nnsum(x), x_s, x_e, label=L"\hat{G}_{NNsum}",
# 	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
# 	plot!(x->Gd_rot(x), x_s, x_e, label=L"\hat{G}_{RotInv}",
# 	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
#     plot!(x->Gd_grad(x), x_s, x_e, label=L"\hat{G}_{\nabla P}",
# 	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
# 	plot!(x->Gd_eos(x), x_s, x_e, label=L"\hat{G}_{EoS}",
# 	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
# 	plot!(x->Gd_phys(x), x_s, x_e, marker=:x, markersize=7,
# 	      markercolor = :black, label=L"\hat{G}_{PhysInf}",
# 	      linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
# 	plot!(x->Gd_gt(x), x_s, x_e, label=L"G_{truth}",
# 		  linewidth = 2.5, legendfontsize=10, color="indigo")
#
#
# 	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=20)
#     # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
#     xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"G", ytickfontsize=14, yguidefontsize=20)
#
#     display(plt)
#     out_path = "./learned_figures/Gd_trained_comp_$(loss_method)_$(method)_kde_D$(D)_N$(N)_θ$(θ)_t$(T).png"
#
#     savefig(plt, out_path)
# end
#
# comparing_Gd(2.7)
# comparing_Gd_log(2.7)
