using Plots, NPZ, LaTeXStrings


T_load = 248;

T = 30;
t_eddy_f = 0.0025 * T;

l_method = "lf"

# m_phys = ["phys_inf_theta", "phys_inf_theta_livescu_ext", "phys_inf_theta_correct_Pi", "phys_inf_Wab_theta",
#                         "phys_inf_W2ab_theta", "phys_inf_Wliu_theta", "phys_inf_theta_po", "phys_inf_Wab_po_theta"];
# #
m_phys = ["phys_inf_theta", "phys_inf_theta_livescu_ext", "phys_inf_theta_correct_Pi",
          "phys_inf_W2ab_theta", "phys_inf_Wliu_theta", "phys_inf_theta_po", "phys_inf_Wab_po_theta"];

#
# m_phys_l = [L"W_{c}" L"W_{c;\theta, l}" L"W_{c, \Pi}" L"W_{ab}" L"W_{ab, \theta}" L"W_{2ab,\theta}" L"W_{q,\theta}" L"W_{c,\theta, p_0}" L"W_{ab,\theta, p_0}"];
m_phys_l = [L"W_{c}" L"W_{c;\theta, l}" L"W_{c, \Pi}" L"W_{ab}" L"W_{2ab,\theta}" L"W_{q,\theta}" L"W_{c,\theta, p_0}" L"W_{ab,\theta, p_0}"];

m_nn_comp = ["node_norm", "nnsum2_norm_theta", "rot_inv", "grad_p_theta", "eos_nn_theta", "phys_inf_Wab_po_theta"];

#
methods_nn = [L"NODE" L"\sum NN" L"RotInv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{ab; p_0, \theta}"];
methods_nn_bar = [L"NODE", L"\sum NN", L"RotInv", L"(\nabla P)_{nn}", L"P_{nn}", L"W_{ab; p_0, \theta}"];


# m_all = m_nn_comp; m_all_l = methods_nn;
m_all = m_phys; m_all_l = m_phys_l;

m_kind = "phys"

t_start = 1;
h = 0.335;
h_kde = 0.9
t_coarse = 2
dt = t_coarse*0.02;


include("./data_loader.jl")
pos_path = "./wc_dns_4096_unif_longer_time/pos_traj_4k.npy"
vel_path = "./wc_dns_4096_unif_longer_time/vel_traj_4k.npy"
rho_path = "./wc_dns_4096_unif_longer_time/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)

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
	Vel_inc = npzread("./learned_generalization/vel_inc_pred_$(method)_t$(T_load)_$(l_method).npy")
	return Vel_inc
end
function load_diff(method)
	diff = npzread("./learned_generalization/diff_pred_$(method)_t$(T_load)_$(l_method).npy")
	return diff
end
function load_accel(method)
	accl = npzread("./learned_data/accl_N4096_T250_h0.335_dns_$(method)_$(l_method).npy")
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
	plot!(x->G_m(x, 5), x_s, x_e, label=m_all_l[5],
	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
	plot!(x->G_m(x, 6), x_s, x_e, marker=:x, markersize=7,
	      markercolor = :black, label=m_all_l[6],
	      linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
	plot!(x->G_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=14, xguidefontsize=20)
	ylabel!(L"log(G_{\delta V})", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Gv_trained_comp_log_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end



function comparing_GV(width=1.2)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
    plt = plot(x->G_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	plot!(x->G_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
	plot!(x->G_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->G_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
	plot!(x->G_m(x, 5), x_s, x_e, label=m_all_l[5],
	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
	plot!(x->G_m(x, 6), x_s, x_e, marker=:x, markersize=7,
	      markercolor = :black, label=m_all_l[6],
	      linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
	plot!(x->G_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")


	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"G_{\delta V}", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Gv_trained_comp_$(m_kind)_t$(T).png"

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
	plt = plot(x->Gd_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
	plot!(x->Gd_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
	plot!(x->Gd_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
    plot!(x->Gd_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
	plot!(x->Gd_m(x, 5), x_s, x_e, label=m_all_l[5],
	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
	plot!(x->Gd_m(x, 6), x_s, x_e, marker=:x, markersize=7,
	      markercolor = :black, label=m_all_l[6],
	      linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
	plot!(x->Gd_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=14, xguidefontsize=20)
	ylabel!(L"log(G_d)", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Gd_trained_comp_log_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end


function comparing_Gd(width=1.2)
    gr(size=(700,600))
    x_s = -0.01
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Gd_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	plot!(x->Gd_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
	plot!(x->Gd_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->Gd_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
	plot!(x->Gd_m(x, 5), x_s, x_e, label=m_all_l[5],
	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
	plot!(x->Gd_m(x, 6), x_s, x_e, marker=:x, markersize=7,
	      markercolor = :black, label=m_all_l[6],
	      linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
	plot!(x->Gd_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")


	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"G_d", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Gd_trained_comp_$(m_kind)_t$(T).png"

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
	accl_m = zeros(T, N, D, n_methods)
	ii = 1;
	for m in m_all
		accl_m[:,:,:,ii] = load_vel_inc(m)
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
	plt = plot(x->Ga_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
	plot!(x->Ga_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
	plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5],
	  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
	plot!(x->Ga_m(x, 6), x_s, x_e, marker=:x, markersize=7,
	      markercolor = :black, label=m_all_l[6],
	      linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
	ylabel!(L"log(G_a)", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Ga_trained_comp_log_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end


function comparing_Ga(width=1.2)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Ga_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	plot!(x->Ga_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="blue")
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
	plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5],
	  	  linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="gold")
	plot!(x->Ga_m(x, 6), x_s, x_e, marker=:x, markersize=7,
	      markercolor = :black, label=m_all_l[6],
	      linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")


	title!(L"\textrm{Learned - Distributions: } t \sim %$t_eddy_f t_{eddy}", titlefont=20)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Acceleration - Statistic } ", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Ga_trained_comp_$(m_kind)_t$(T).png"

    savefig(plt, out_path)
end



gv_w = T * 0.007
gv_w2 = T * 0.005
comparing_GV(gv_w)
comparing_GV_log(gv_w2)

gd_w = T * 0.004;
gd_w2 = T * 0.003;
comparing_Gd(gd_w)
comparing_Gd_log(gd_w2)


comparing_Ga(0.4)
comparing_Ga_log(0.21)






"""
==============================================================
Animating Distributions
===============================================================
"""

Gat_m(x, t, m_n) = kde(x, vec(accl_m[t, :, :, m_n]))
Gat_gt(x, t) = kde(x, vec(accl_gt[t, :, :]));

function animate_dist_log(width=0.21, sim_time = 30)
	sim_path = "./learned_sims/animate_dist_accl_log.mp4"
	x_s = -width; x_e = width;
	anim = @animate for t ∈ 25:T
		println("time step = ", t); t_s = round(t*dt, digits=1)
		plt = plot(x->Gat_m(x, t, 1), x_s, x_e, label=m_all_l[1],
	          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
		plot!(x->Gat_m(x, t, 2), x_s, x_e, label=m_all_l[2],
		  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="blue")
		plot!(x->Gat_m(x, t, 3), x_s, x_e, label=m_all_l[3],
		      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
	    plot!(x->Gat_m(x, t, 4), x_s, x_e, label=m_all_l[4],
		  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
		plot!(x->Gat_m(x, t, 5), x_s, x_e, label=m_all_l[5],
		  	  linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="gold")
		plot!(x->Gat_m(x, t, 6), x_s, x_e, marker=:x, markersize=7,
		      markercolor = :black, label=m_all_l[6],
		      linestyle=:dot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="grey40")
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

# animate_dist_log(0.22)


function animate_dist(width=0.21, sim_time = 30)
	sim_path = "./learned_sims/animate_dist_accl.mp4"
	x_s = -width; x_e = width;
	anim = @animate for t ∈ 5:T
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
		plot!(x->Gat_m(x, t, 6), x_s, x_e, marker=:x, markersize=7,
		      markercolor = :black, label=m_all_l[6],
		      linestyle=:dot, linewidth = 2.5, legendfontsize=10, color="grey40")
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

# animate_dist(0.4)
