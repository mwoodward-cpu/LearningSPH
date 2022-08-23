using Plots, NPZ, LaTeXStrings


# T_load = 150;
T_load = 500;
T = 400;

T_pred = 500; T_train = 50;
t_eddy_f = 0.0025 * T;

l_method = "lf_klt"
IC = "dns_equil"

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
# m_tot = vcat(m_phys, m_nn);
m_comb = vcat(m_nns, m_phys[4], m_phys[1])

methods_phys = [L"W_{2}(a,b)" L"W(a,b)" L"W_{cub}" L"W_{quart}"];

#
methods_nn = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}"] # L"W_{ab; p_0, \theta}"];
methods_comb = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{cub}" L"W_{2}(a,b)"] # L"W_{ab; p_0, \theta}"];



# m_all = m_nns; m_all_l = methods_nn;
# m_all = m_phys; m_all_l = methods_phys;
m_all = m_comb; m_all_l = methods_comb;

# m_kind = "phys"
# m_kind = "nn_comp"
m_kind = "comp"

Mt = 0.08;
t_start = 1;
h = 0.335;
h_kde = 0.9
t_coarse = 1
dt = t_coarse*0.04;


include("./data_loader.jl")
pos_path = "./equil_ic_data/mt008/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt008/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt008/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;

include("kde_G.jl")

G_gt(x) = kde(x, vec(Vel_inc_gt[T, :, :]));
Gd_gt(x) = kde(x, vec(Diff_gt[T, :]));

function load_vel_inc(method)
	Vel_inc = npzread("./learned_generalization_t50/vel_inc_pred_$(method)_t$(T_load)_$(l_method)_Mt$(Mt).npy")
	return Vel_inc
end
function load_diff(method)
	diff = npzread("./learned_generalization_t50/diff_pred_$(method)_t$(T_load)_$(l_method)_Mt$(Mt).npy")
	return diff
end
function load_accel(method)
	acc_path = "./learned_data_t50/accl_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	accl = npzread(acc_path)
	return accl
end

function load_all_vel_inc(m_all, T)
	n_methods = size(m_all)[1];
	vel_inc_m = zeros(T, N, D, n_methods)
	ii = 1;
	for m in m_all
		vel_inc_m[:,:,:,ii] = load_vel_inc(m)
		ii += 1;
	end
	return vel_inc_m
end

vel_inc_m = load_all_vel_inc(m_all, T_load)
G_m(x, m_n) = kde(x, vec(vel_inc_m[T, :, :, m_n]))



function comparing_GV_log(width=1.2)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
    plt = plot(x->G_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
	plot!(x->G_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
	plot!(x->G_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
    plot!(x->G_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
	if m_kind == "nn_comp"
		plot!(x->G_m(x, 5), x_s, x_e, label=m_all_l[5],
		  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
	end
    # plot!(x->G_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:xcross, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="black")
	# plot!(x->G_m(x, 7), x_s, x_e, marker=:x, markersize=7,
	#       markercolor = :black, label=m_all_l[7],
	#       linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
	plot!(x->G_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=20)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=14, xguidefontsize=20)
	ylabel!(L"log(G_{\delta V})", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures_t50/Gv_t_50_trained_comp_log_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end



function comparing_GV(width=1.2)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
    plt = plot(x->G_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	# plot!(x->G_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
	plot!(x->G_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->G_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
	if m_kind == "nn_comp"
		plot!(x->G_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
		  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
		  end
	#
	if m_kind == "comp"
		plot!(x->G_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
		#
		plot!(x->G_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:auto, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
		#
		plot!(x->G_m(x, 7), x_s, x_e, label=m_all_l[7], markershape=:auto, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
	end
	# plot!(x->G_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:xcross, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="black")
	# plot!(x->G_m(x, 7), x_s, x_e, marker=:x, markersize=7,
	#       markercolor = :black, label=m_all_l[7],
	#       linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
	plot!(x->G_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=14, color="indigo")


	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=22)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=16, xguidefontsize=20)
    ylabel!(L"G_{\delta V}", ytickfontsize=16, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures_t50/Gv_t_50_trained_comp_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end



function load_all_diff(m_all, T)
	n_methods = size(m_all)[1];
	diff_m = zeros(T, N, n_methods)
	ii = 1;
	for m in m_all
		diff_m[:,:,ii] = load_diff(m)
		ii += 1;
	end
	return diff_m
end
#

diff_m = load_all_diff(m_all, T_load)
Gd_m(x, m_n) = kde(x, vec(diff_m[T, :, m_n]))



function comparing_Gd_log(width=1.2)
    gr(size=(700,600))
    x_s = -0.001
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Gd_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=12, color="maroon")
	plot!(x->Gd_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = 5.5,
	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=12, color="blue")
	plot!(x->Gd_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=12, color="green")
    plot!(x->Gd_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=12, color="turquoise")
	if m_kind == "nn_comp"
	plot!(x->Gd_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=12, color="gold")
	  end
	#
	# if m_kind == "comp"
	# 	plot!(x->Gd_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
	# 		  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
	# 	#
	# 	plot!(x->Gd_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:auto, ms = 5.5,
	# 		  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
	# 	#
	# 	plot!(x->Gd_m(x, 7), x_s, x_e, label=m_all_l[7], markershape=:auto, ms = 5.5,
	# 		  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
	# end
	# plot!(x->Gd_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:xcross, ms = 5.5,
	# 	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=12, color="black")
	# plot!(x->Gd_m(x, 7), x_s, x_e, marker=:x, markersize=7,
	# 	  markercolor = :black, label=m_all_l[7],
	# 	  linestyle=:dot, linewidth = 2.5,  yaxis=:log, legendfontsize=10, color="grey40")
	plot!(x->Gd_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=14, color="indigo")

	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=22)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=16, xguidefontsize=20)
	ylabel!(L"log(G_{\delta x})", ytickfontsize=16, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures_t50/Gd_t_50_trained_comp_log_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end


function comparing_Gd(width=1.2)
    gr(size=(700,600))
    x_s = -0.01
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Gd_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, legendfontsize=12, color="maroon")
	# plot!(x->Gd_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=12, color="blue")
	plot!(x->Gd_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=12, color="green")
    plot!(x->Gd_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=12, color="turquoise")
	if m_kind == "nn_comp"
	plot!(x->Gd_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=12, color="gold")
	  end
	#
	if m_kind == "comp"
		plot!(x->Gd_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
		#
		plot!(x->Gd_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:auto, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
		#
		plot!(x->Gd_m(x, 7), x_s, x_e, label=m_all_l[7], markershape=:auto, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
	end
	# plot!(x->Gd_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:xcross, ms = 5.5,
	# 	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=12, color="black")
	# plot!(x->Gd_m(x, 7), x_s, x_e, marker=:x, markersize=7,
	# 	  markercolor = :black, label=m_all_l[7],
	# 	  linestyle=:dot, linewidth = 2.5,  legendfontsize=10, color="grey40")
	plot!(x->Gd_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=14, color="indigo")


	title!(L"\textrm{Learned - Distributions: } t \sim t_{eddy}", titlefont=22)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=16, xguidefontsize=20)
    ylabel!(L"G_d", ytickfontsize=16, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures_t50/Gd_t_50_trained_comp_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end



"""
Plotting acceleration stats
"""

function compute_gt_accl(vel_gt)
	accl_gt = zeros(T, N, D);
	for t in 1 : (T-1)
		for i in 1 : D
			accl_gt[t+1,:,i] = (vel_gt[t+1,:,i] .- vel_gt[t,:,i]) / dt
		end
	end
	return accl_gt
end

accl_gt = compute_gt_accl(vels_gt)
Ga_gt(x) = kde(x, vec(accl_gt[T, :, :]));

function load_all_accl(m_all, T)
	n_methods = size(m_all)[1];
	accl_m = zeros(501, N, D, n_methods)
	ii = 1;
	for m in m_all
		accl_m[:,:,:,ii] = load_accel(m)
		ii += 1;
	end
	return accl_m
end

accl_m = load_all_accl(m_all, T_load)
Ga_m(x, m_n) = kde(x, vec(accl_m[T, :, :, m_n]))




function comparing_Ga_log(width=1.2)
    gr(size=(700,600))
    x_s = -width
    x_e = width

	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Ga_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
	# plot!(x->Ga_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
	if m_kind == "nn_comp"
	plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
	  end
	# plot!(x->Ga_m(x, 6), x_s, x_e, label=m_all_l[6],
	# 	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="black")
	# plot!(x->Ga_m(x, 7), x_s, x_e, marker=:x, markersize=7,
	# 	  markercolor = :black, label=m_all_l[7],
	# 	  linestyle=:dot, linewidth = 2.5,  yaxis=:log, legendfontsize=10, color="grey40")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
	ylabel!(L"log(G_a)", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures_t50/Ga_t_50_trained_comp_log_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end


function comparing_Ga(width=1.2)
    gr(size=(700,600))
    x_s = -width
    x_e = width

	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Ga_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	# plot!(x->Ga_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
    if m_kind == "nn_comp"
	plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
	  end
	#
	if m_kind == "comp"
		plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
		#
		plot!(x->Ga_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:auto, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
		#
		plot!(x->Ga_m(x, 7), x_s, x_e, label=m_all_l[7], markershape=:auto, ms = 5.5,
			  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color=:auto)
	end
	# plot!(x->Ga_m(x, 6), x_s, x_e, label=m_all_l[6],
	#   		  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="black")
	# plot!(x->Ga_m(x, 7), x_s, x_e, marker=:x, markersize=7,
	#   		  markercolor = :black, label=m_all_l[7],
	#   		  linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")

		  #%$t_eddy_f
	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures_t50/Ga_t_50_trained_comp_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end



gv_w = T * 0.0018
gv_w2 = T * 0.0012
comparing_GV(0.4)
# comparing_GV_log(gv_w2)
#
gd_w = T^(1.35) * 0.0014;
comparing_Gd(gd_w)
# comparing_Gd_log(gd_w)



comparing_Ga(0.1)
# comparing_Ga_log(0.085)






"""
==============================================================
Animating Distributions
===============================================================
"""

Gat_m(x, t, m_n) = kde(x, vec(accl_m[t, :, :, m_n]))
Gat_gt(x, t) = kde(x, vec(accl_gt[t, :, :]));

function animate_dist_log(width=0.21, sim_time = 30)
	sim_path = "./learned_sims/animate_dist_accl_log_$(m_kind)_t$(T).mp4"
	x_s = -width; x_e = width;
	anim = @animate for t ∈ 25:T
		println("time step = ", t); t_s = round(t*dt, digits=1)
		# plt = plot(x->Gat_m(x, t, 1), x_s, x_e, label=m_all_l[1],
	    #       linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
		# plot!(x->Gat_m(x, t, 2), x_s, x_e, label=m_all_l[2],
		#   	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
		# plot!(x->Gat_m(x, t, 3), x_s, x_e, label=m_all_l[3],
		#       linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
	    # plot!(x->Gat_m(x, t, 4), x_s, x_e, label=m_all_l[4],
		#   	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
		plt = plot(x->Gat_m(x, t, 5), x_s, x_e, label=m_all_l[5],
		  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
		plot!(x->Gat_m(x, t, 6), x_s, x_e, label=m_all_l[6],
					linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="black")
		# plot!(x->Gat_m(x, t, 7), x_s, x_e, marker=:x, markersize=7,
		#       markercolor = :black, label=m_all_l[7],
		#       linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
		plot!(x->Gat_gt(x, t), x_s, x_e, label=L"G_{truth}",
			  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

		title!(L"\textrm{Learned - Distributions: } t = %$t_s (s)", titlefont=20)
		      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
		xlabel!(L"\frac{dv}{dt}", xtickfontsize=14, xguidefontsize=20)
		ylabel!(L"\textrm{Acceleration PDF}", ytickfontsize=14, yguidefontsize=20)
	end
	gif(anim, sim_path, fps = round(Int, T/sim_time))
	println("****************  Simulation COMPLETE  *************")
end

# animate_dist_log(0.08)


function animate_dist(width=0.21, sim_time = 30)
	sim_path = "./learned_sims_t50/animate_dist_accl_$(m_kind)_t$(T).mp4"
	x_s = -width; x_e = width;
	anim = @animate for t ∈ 25:T
		println("time step = ", t); t_s = round(t*dt, digits=1)
		plt = plot(x->Gat_m(x, t, 1), x_s, x_e, label=m_all_l[1],
	          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
		plot!(x->Gat_m(x, t, 2), x_s, x_e, label=m_all_l[2],
		  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
		plot!(x->Gat_m(x, t, 3), x_s, x_e, label=m_all_l[3],
		      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
	    plot!(x->Gat_m(x, t, 4), x_s, x_e, label=m_all_l[4],
		  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
		plot!(x->Gat_m(x, t, 5), x_s, x_e, label=m_all_l[5],
		  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
		plot!(x->Gat_m(x, t, 6), x_s, x_e, label=m_all_l[6],
					linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="black")
		# plot!(x->Gat_m(x, t, 7), x_s, x_e, marker=:x, markersize=7,
		#       markercolor = :black, label=m_all_l[7],
		#       linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
		plot!(x->Gat_gt(x, t), x_s, x_e, label=L"G_{truth}",
			  linewidth = 2.5, legendfontsize=10, color="indigo")

		title!(L"\textrm{Learned - Distributions: } t = %$t_s (s)", titlefont=20)
		      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
		xlabel!(L"\frac{dv}{dt}", xtickfontsize=14, xguidefontsize=20)
		ylabel!(L"\textrm{Acceleration PDF}", ytickfontsize=14, yguidefontsize=20)
	end
	gif(anim, sim_path, fps = round(Int, T/sim_time))
	println("****************  Simulation COMPLETE  *************")
end

# animate_dist(0.1)

function obtain_ga_over_all_t(width=0.1)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	Gat2_m(x, m_n) = kde(x, vec(accl_m[2:end, :, :, m_n]))
	Gat2_gt(x) = kde(x, vec(accl_gt[2:end, :, :]));
	plt = plot(x->Gat2_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, legendfontsize=16, color="maroon")
	plot!(x->Gat2_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = 5.5,
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=16, color="blue")
	plot!(x->Gat2_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=16, color="green")
    plot!(x->Gat2_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=16, color="turquoise")
    if m_kind == "nn_comp"
	plot!(x->Gat2_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = 5.5,
	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=16, color="gold")
	  end
	# plot!(x->Gat2_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:xcross, ms = 5.5,
	#   	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=16, color="black")
	plot!(x->Gat2_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=16, color="indigo")

		  title!(L"\textrm{Learned Acceleration over t}", titlefont=26)
	      xlabel!(L"\textrm{Acceleration} ", xtickfontsize=18, xguidefontsize=24)
	      ylabel!(L"G_a", ytickfontsize=18, yguidefontsize=24)
    display(plt)
	out_path = "./learned_figures_t50/Ga_t_50_comb_t_comp_$(m_kind)_t$(T).png"

	savefig(plt, out_path)
end
# obtain_ga_over_all_t()


function obtain_ga_log_over_all_t(width=0.1)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	Gat2_m(x, m_n) = kde(x, vec(accl_m[2:end, :, :, m_n]))
	Gat2_gt(x) = kde(x, vec(accl_gt[2:end, :, :]));
	plt = plot(x->Gat2_m(x, 1), x_s, x_e, label=m_all_l[1], yaxis=:log, markershape=:heptagon, ms = 5.5,
          linestyle=:dash, linewidth = 2.5, legendfontsize=16, color="maroon")
	plot!(x->Gat2_m(x, 2), x_s, x_e, label=m_all_l[2], yaxis=:log, markershape=:star8, ms = 5.5,
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=16, color="blue")
	plot!(x->Gat2_m(x, 3), x_s, x_e, label=m_all_l[3], yaxis=:log, markershape=:ltriangle, ms = 5.5,
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=16, color="green")
    plot!(x->Gat2_m(x, 4), x_s, x_e, label=m_all_l[4], yaxis=:log, markershape=:diamond, ms = 5.5,
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=16, color="turquoise")
    if m_kind == "nn_comp"
	plot!(x->Gat2_m(x, 5), x_s, x_e, label=m_all_l[5], yaxis=:log, markershape=:rect, ms = 5.5,
		  linestyle=:dash, linewidth = 2.5, legendfontsize=16, color="gold")
	end
	plot!(x->Gat2_gt(x), x_s, x_e, label=L"G_{truth}", yaxis=:log,
	  		linewidth = 2.5, legendfontsize=16, color="indigo")


		  title!(L"\textrm{Learned Acceleration over t}", titlefont=26)
	      xlabel!(L"\textrm{Acceleration} ", xtickfontsize=18, xguidefontsize=24)
	      ylabel!(L"log(G_a)", ytickfontsize=18, yguidefontsize=24)
    display(plt)
	out_path = "./learned_figures_t50/Ga_t_50_log_comb_t_comp_$(m_kind)_t$(T).png"

	savefig(plt, out_path)
end
# obtain_ga_log_over_all_t(0.09)
