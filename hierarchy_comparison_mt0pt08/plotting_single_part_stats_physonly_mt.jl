using Plots, NPZ, LaTeXStrings


T_load = 30;
T = 30;

# for Mt in [0.04, 0.08, 0.16];
Mt = 0.16;

t_eddy_f = 0.0025 * T;

l_method = "lf"

m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi",
          "phys_inf_Wliu_theta_po_liv_Pi"];

m_nn = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		  "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi"];
m_tot = vcat(m_phys, m_nn);

methods_phys = [L"W_{cub}" L"W_{a, b}" L"W2_{a,b}" L"W_{quart}"];
# methods_phys_bar = [L"W_{cub}", L"W_{a, b}"];

#
methods_nn = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{ab; p_0, \theta}"];
methods_nn_bar = [L"NODE", L"\sum NN", L"Rot-Inv", L"(\nabla P)_{nn}", L"P_{nn}", L"W_{ab; p_0, \theta}"];



# m_all = m_nn; m_all_l = methods_nn;
m_all = m_phys; m_all_l = methods_phys

m_kind = "phys"


t_start = 1;
h = 0.335;
h_kde = 0.9
t_coarse = 2
dt = t_coarse*0.02;


include("./data_loader.jl")
pos_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/pos_traj_4k.npy"
vel_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/vel_traj_4k.npy"
rho_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/rho_traj_4k.npy"
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

function load_vel_inc(method, mt)
	Vel_inc = npzread("./learned_generalization/vel_inc_pred_$(method)_t$(T_load)_$(l_method)_Mt$(mt).npy")
	return Vel_inc
end
function load_diff(method, mt)
	diff = npzread("./learned_generalization/diff_pred_$(method)_t$(T_load)_$(l_method)_Mt$(mt).npy")
	return diff
end
function load_accel(method, itr, lr, mt)
	acc_path = "./learned_data/accl_Tp30_Tt30_h0.335_dns_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(mt).npy"
	accl = npzread(acc_path)
	return accl
end
function obtain_itr_lr(method)
	if method =="phys_inf_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="phys_inf_Wab_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="phys_inf_W2ab_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="phys_inf_Wliu_theta_po_liv_Pi"
		itr = 2200; lr = 0.02;
	end
	if method =="node_norm_theta_liv_Pi"
		itr = 500; lr = 0.002;
	end
	if method =="nnsum2_norm_theta_liv_Pi"
		itr = 600; lr = 0.01;
	end
	if method =="rot_inv_theta_liv_Pi"
		itr = 500; lr = 0.02;
	end
	if method =="grad_p_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02;
	end
	if method =="eos_nn_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02;
	end
	return itr, lr
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
	accl_m = zeros(31, N, D, n_methods)
	ii = 1;
	for m in m_all
		itr, lr = obtain_itr_lr(m)
		accl_m[:,:,:,ii] = load_accel(m, itr, lr, Mt)
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
	plt = plot(x->Ga_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
	plot!(x->Ga_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="black")
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

		  title!(L"\textrm{Learned - Distributions: } M_t = %$Mt", titlefont=20)
	      # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
	      xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
	      ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Ga_trained_comp_log_$(m_kind)_t$(T)_mt$(Mt).png"

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
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="black")
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")


	title!(L"\textrm{Learned - Distributions: } M_t = %$Mt", titlefont=20)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    out_path = "./learned_figures/Ga_trained_comp_$(m_kind)_t$(T)_mt$(Mt).png"

    savefig(plt, out_path)
end


if Mt == 0.04
	comparing_Ga(0.02)
	comparing_Ga_log(0.019)
end

if Mt == 0.08
	comparing_Ga(0.1)
	comparing_Ga_log(0.07)
end

if Mt == 0.16
	comparing_Ga(0.2)
	comparing_Ga_log(0.18)
end



function obtain_ga_over_all_t(width=0.1)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	Gat2_m(x, m_n) = kde(x, vec(accl_m[2:end, :, :, m_n]))
	Gat2_gt(x) = kde(x, vec(accl_gt[2:end, :, :]));
	plt = plot(x->Gat2_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	plot!(x->Gat2_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="black")
	plot!(x->Gat2_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, legendfontsize=10, color="green")
    plot!(x->Gat2_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, legendfontsize=10, color="turquoise")
	plot!(x->Gat2_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")

		  title!(L"\textrm{Learned - Distributions over all t: } M_t = %$Mt", titlefont=20)
	      xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
	      ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)
    display(plt)
	out_path = "./learned_figures/Ga_comb_t_comp_$(m_kind)_t$(T)_mt$(Mt).png"

	savefig(plt, out_path)
end


function obtain_ga_log_over_all_t(width=0.1)
    gr(size=(700,600))
    x_s = -0.105
    x_e = 0.115
	Gat2_m(x, m_n) = kde(x, vec(accl_m[2:end, :, :, m_n]))
	Gat2_gt(x) = kde(x, vec(accl_gt[2:end, :, :]));
	plt = plot(x->Gat2_m(x, 1), x_s, x_e, label=m_all_l[1],
          linestyle=:dash, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="maroon")
	plot!(x->Gat2_m(x, 2), x_s, x_e, label=m_all_l[2],
	  	  linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="black")
	plot!(x->Gat2_m(x, 3), x_s, x_e, label=m_all_l[3],
	      linestyle=:dashdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="green")
    plot!(x->Gat2_m(x, 4), x_s, x_e, label=m_all_l[4],
	  	  linestyle=:dashdotdot, linewidth = 2.5, yaxis=:log, legendfontsize=10, color="turquoise")
	plot!(x->Gat2_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, yaxis=:log, legendfontsize=10, color="indigo")

		  title!(L"\textrm{Learned - Distributions over all t: } M_t = %$Mt", titlefont=20)
	      xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
	      ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)
    display(plt)
	out_path = "./learned_figures/Ga_log_comb_t_comp_$(m_kind)_t$(T)_mt$(Mt).png"

	savefig(plt, out_path)
end

if Mt == 0.16
	obtain_ga_over_all_t(0.2)
	obtain_ga_log_over_all_t(0.18)
end

if Mt == 0.04
	obtain_ga_over_all_t(0.02)
	obtain_ga_log_over_all_t(0.012)
end

if Mt == 0.08
	obtain_ga_over_all_t(0.1)
	obtain_ga_log_over_all_t(0.05)
end
