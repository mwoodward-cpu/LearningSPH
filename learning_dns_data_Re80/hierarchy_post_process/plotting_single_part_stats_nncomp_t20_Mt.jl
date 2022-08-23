using Plots, NPZ, LaTeXStrings

T_load = 20;
T = 20; T_2 = 20;

T_pred = 200; T_train = 20;
t_eddy_f = 0.0025 * T;
# T_frame = 1:200; #1:50 good
T_frame = 100:200; #over transitory for mt016;


l_method = "lf"
IC = "dns_equil"


# Mt = 0.08; mmt = "008"
Mt = 0.16; mmt = "016"
#Mt = 0.04; mmt = "004"

include("plot_dims.jl")



m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wliu_theta_po_liv_Pi",
		"phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
# m_tot = vcat(m_phys, m_nn);
m_comb = vcat(m_nns, m_phys)

methods_phys = [L"W_{cub}" L"W_{quart}" L"W(a,b)" L"W_{2}(a,b)"];

#
methods_nn = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}"] # L"W_{ab; p_0, \theta}"];
methods_comb = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{cub}" L"W_{quart}" L"W(a,b)" L"W_{2}(a,b)"] # L"W_{ab; p_0, \theta}"];


# m_kind = "nn_comp"; m_all = m_nns; m_all_l = methods_nn;
# m_kind = "phys"; m_all = m_phys; m_all_l = methods_phys;
# m_kind = "comb"; m_all = m_comb; m_all_l = methods_comb;
m_kind = "comb";


if m_kind == "phys" m_all = m_phys; m_all_l = methods_phys; end
if m_kind == "comb" m_all = m_comb; m_all_l = methods_comb; end


#colors of plots
include("./color_scheme_utils.jl")
cs1=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
cs2=[c6,c7,c8,c9];
cs3=[c1,c2,c3,c4,c5];
if m_kind == "comb" cs = cs1; end
if m_kind == "phys" cs = cs2; end




t_start = 1;
h = 0.335;
h_kde = 0.9
t_coarse = 1
dt = t_coarse*0.04;


include("./data_loader.jl")
pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
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
	Vel_inc = npzread("./learned_generalization_mt_rel/vel_inc_pred_$(method)_t$(T_load)_tp$(T_pred)_$(l_method)_Mt$(Mt).npy")
	return Vel_inc
end
function load_diff(method)
	diff = npzread("./learned_generalization_mt_rel/diff_pred_$(method)_t$(T_load)_tp$(T_pred)_$(l_method)_Mt$(Mt).npy")
	return diff
end
function load_accel(method)
	acc_path = "./learned_data_mt_relation/accl_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	accl = npzread(acc_path)
	return accl
end
function load_accel_mt(method)
	acc_path = "./learned_data_mt_relation/accl_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
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





"""
Plotting acceleration stats
"""

function compute_gt_accl(vel_gt)
	accl_gt = zeros(T_pred, N, D);
	for t in 1 : (T_pred)
		for i in 1 : D
			accl_gt[t,:,i] = (vel_gt[t+1,:,i] .- vel_gt[t,:,i]) / dt
		end
	end
	return accl_gt
end

accl_gt = compute_gt_accl(vels_gt)
Ga_gt(x) = kde(x, vec(accl_gt[T_frame, :, :]));

function load_all_accl(m_all, T)
	n_methods = size(m_all)[1];
	accl_m = zeros(T_pred+1, N, D, n_methods)
	ii = 1;
	for m in m_all
		if Mt==0.08
			accl_m[:,:,:,ii] = load_accel(m)
		else
			accl_m[:,:,:,ii] = load_accel_mt(m)
		end
		ii += 1;
	end
	return accl_m
end

accl_m = load_all_accl(m_all, T_load)
Ga_m(x, m_n) = kde(x, vec(accl_m[T_frame, :, :, m_n]))


function comparing_G_all(G, Gin_gt, yaxis_log, stat_obj, width=1.2)
	x_s = -width
	x_e = width
	lw_gt = 9.5;
	if stat_obj=="dx" x_s = -0.01 end

	plt = plot(x->G(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = ms_,
		  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[1])
	plot!(x->G(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = ms_,
		  linestyle=:dashdot, linewidth = lw, legendfontsize=10, color=cs[2])
	plot!(x->G(x, 3), x_s, x_e, label=m_all_l[3], markershape=:circle, ms = ms_,
		  linestyle=:dashdot, linewidth = lw, legendfontsize=10, color=cs[3])
	plot!(x->G(x, 4), x_s, x_e, label=m_all_l[4], markershape=:auto, ms = ms_,
		  linestyle=:dashdotdot, linewidth = lw, legendfontsize=10, color=cs[4])
	if m_kind == "nn_comp"
		plot!(x->G(x, 5), x_s, x_e, label=m_all_l[5], markershape=:xcross, ms = ms_,
		  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[5])
	end
	if m_kind == "comb"
		plot!(x->G(x, 5), x_s, x_e, label=m_all_l[5], markershape=:auto, ms = ms_,
			  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[5])
		#
		plot!(x->G(x, 6), x_s, x_e, label=m_all_l[6], markershape=:auto, ms = ms_,
			  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[6])
		#
		plot!(x->G(x, 7), x_s, x_e, label=m_all_l[7], markershape=:auto, ms = ms_,
			  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[7])
		#
		plot!(x->G(x, 8), x_s, x_e, label=m_all_l[8], markershape=:auto, ms = ms_,
			  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[8])
		#
		plot!(x->G(x, 9), x_s, x_e, label=m_all_l[9], markershape=:rect, ms = ms_,
			  linestyle=:dash, linewidth = lw, legendfontsize=10, color=cs[9])
		#
	end

	plot!(x->Gin_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = lw_gt, legendfontsize=legend_fs, color="black")

	title!(L"\textrm{Learned ~ Distributions: }  M_t = %$Mt", titlefont=title_fs)
	if stat_obj=="dv"
		xlabel!(L"\textrm{Velocity - increment: }  \delta V", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
	    ylabel!(L"PDF_{\delta v}", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

	    display(plt)
	    out_path = "./learned_figures_mt_rel/Gv_trained_comp_$(m_kind)_t$(T)_Mt$(Mt).png"
		savefig(plt, out_path)
	end
	if stat_obj=="dx"
		xlabel!(L"\textrm{Dispersion - Statistic } ", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
	    ylabel!(L"PDF_{\delta x}", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

	    display(plt)
	    out_path = "./learned_figures_mt_rel/Gd_trained_comp_$(m_kind)_t$(T)_Mt$(Mt).png"

	    savefig(plt, out_path)
	end
	if stat_obj=="a"
		xlabel!(L"\textrm{Acceleration} ", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
		ylabel!(L"PDF_a", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

		display(plt)
		out_path = "./learned_figures_mt_rel/Ga_trained_comp_$(m_kind)_t$(T)_Mt$(Mt).png"

		savefig(plt, out_path)
	end
end


function comparing_Ga_log(width=1.2)
    x_s = -width
    x_e = width
	lw_gt = 9.5;

	plt = plot(x->Ga_m(x, 1), x_s, x_e, label=m_all_l[1], markershape=:heptagon, ms = ms_,
          linestyle=:dash, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[1])
	plot!(x->Ga_m(x, 2), x_s, x_e, label=m_all_l[2], markershape=:star8, ms = ms_,
	  	  linestyle=:dashdot, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[2])
	plot!(x->Ga_m(x, 3), x_s, x_e, label=m_all_l[3], markershape=:circle, ms = ms_,
	      linestyle=:dashdot, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[3])
    plot!(x->Ga_m(x, 4), x_s, x_e, label=m_all_l[4], markershape=:rect, ms = ms_,
	  	  linestyle=:dashdotdot, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[4])
if m_kind == "nn_comp"
	plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:xcross, ms = ms_,
	  	  linestyle=:dash, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[5])
	  end
	if m_kind == "comb"
  		plot!(x->Ga_m(x, 5), x_s, x_e, label=m_all_l[5], markershape=:rect, ms = ms_,
  		  	  linestyle=:dash, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[5])
		#
		plot!(x->Ga_m(x, 6), x_s, x_e, label=m_all_l[6], markershape=:auto, ms = ms_,
			  linestyle=:dash, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[6])
		#
		plot!(x->Ga_m(x, 7), x_s, x_e, label=m_all_l[7], markershape=:auto, ms = ms_,
			  linestyle=:dash, linewidth = lw, yaxis=:log, legendfontsize=10, color=cs[7])

    end
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = lw_gt, yaxis=:log, legendfontsize=legend_fs, color="black")

	title!(L"\textrm{Learned ~ Distributions: }  M_t = %$Mt", titlefont=title_fs)
	xlabel!(L"\textrm{Acceleration} ", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
	ylabel!(L"log(G_a)", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

    display(plt)
    out_path = "./learned_figures_mt_rel/Ga_trained_comp_log_$(m_kind)_t$(T)_Mt$(Mt).png"

    savefig(plt, out_path)
end

if Mt==0.16
	winv = 0.05; wind = 0.04;
#	comparing_G_all(G_m, G_gt, 0, "dv", winv)
#	comparing_G_all(Gd_m, Gd_gt, 0, "dx", wind)
end
if Mt==0.04
	winv = 0.005; wind = 0.004
#	comparing_G_all(G_m, G_gt, 0, "dv", winv)
#	comparing_G_all(Gd_m, Gd_gt, 0, "dx", wind)
end


if Mt==0.04
	win = 0.042
end

if Mt==0.08
	win = 0.2
end

if Mt==0.16
	win = 0.42
end

comparing_G_all(Ga_m, Ga_gt, 0, "a", win)

#if m_kind=="comb"
#	comparing_Ga_log(win/2)
#else comparing_Ga_log(win)
#end
