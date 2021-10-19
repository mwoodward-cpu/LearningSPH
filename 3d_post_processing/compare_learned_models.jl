using Statistics, LaTeXStrings
using NPZ, Plots, Flux
using Flux.Losses
using BSON: @load

function save_output_data(data, path)
    npzwrite(path, data)
end
function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end


loss_method = "kl_lf"; lg_method = "lf";

# method = "node"; latex_method = "NODE"
# method = "nnsum"; latex_method = "NN Sum"
# method = "rot_inv"; latex_method = "Rot Inv"
method = "eos_nn"; latex_method = "EoS NN"
# method = "grad_p"; latex_method = "Grad P"
# method = "Wnn"; latex_method = "W NN"
# method = "phys_inf"; latex_method = "Phys Inf"
# method = "truth"; latex_method = "Truth"


α = 1.0
β = 2.0*α  #usual value of params for alpha and β but these depend on problem
θ = 5e-1;
c = 10.0
g = 7.0
h = 0.335
cdt = 0.4;
dt = cdt * h / c;
h_kde = 0.9;
r = 5.0;
n_int = 200;
T = 120;

ENV["GKSwstype"]="100"
function load_data_files(pos_path, vel_path, rho_path)
    println(" ****************** Loading $(latex_method) data files ***********************")
	traj = npzread(pos_path)
	vels = npzread(vel_path)
	rhos = npzread(rho_path)
	return traj, vels, rhos
end

if method!="truth"
	pos_path = "./learned_data/traj_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_$(method).npy"
	vel_path = "./learned_data/vels_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_$(method).npy"
	rho_path = "./learned_data/rhos_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_$(method).npy"
	traj, vels, rhos = load_data_files(pos_path, vel_path, rho_path)
	D = size(traj)[3];
	N = size(traj)[2];
end




function obtain_snapshots2(traj, n_snaps=4)
	m_s = 3.5
	ratio = 1/(n_snaps+2); horz = 1500; vert = ceil(Int, ratio*horz);
	# gr(size=(horz,vert))
	gr(size=(600,600))

	n_2 = round(Int,N/2);
	t_steps = size(traj)[1]; T = t_steps - 1;
	t_range = ceil.(Int, range(100, ceil(Int, T), length=n_snaps));

	p0 = plot(zeros(2,2), xlims = [0, 1], ylims = [0,1], axis=([], false), grid = false)

	p1 =Plots.scatter(traj[1, 1:n_2, 1], traj[1, 1:n_2, 2], traj[1, 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)#,
		# zlabel=L"\textrm{%$latex_method}", ztickfontsize=1, zguidefontsize=22)
		# zlabel=L"d_t v_i = NN_{\theta}", ztickfontsize=1, zguidefontsize=30)
		Plots.scatter!(traj[1, (n_2+1):end, 1], traj[1, (n_2+1):end, 2],
		traj[1,(n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_0")


	p2 =Plots.scatter(traj[t_range[1], 1:n_2, 1], traj[t_range[1], 1:n_2, 2], traj[t_range[1], 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
		Plots.scatter!(traj[t_range[1], (n_2+1):end, 1], traj[t_range[1], (n_2+1):end, 2],
		traj[t_range[1],(n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_{\lambda}")

	p3 =Plots.scatter(traj[t_range[2], 1:n_2, 1], traj[t_range[2], 1:n_2, 2], traj[t_range[2], 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
		Plots.scatter!(traj[t_range[2], (n_2+1):end, 1], traj[t_range[2], (n_2+1):end, 2],
		traj[t_range[2], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_2")

	p4 =Plots.scatter(traj[t_range[3], 1:n_2, 1], traj[t_range[3], 1:n_2, 2], traj[t_range[3], 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
		Plots.scatter!(traj[t_range[3], (n_2+1):end, 1], traj[t_range[3], (n_2+1):end, 2],
		traj[t_range[3], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_3")

	p5 =Plots.scatter(traj[t_range[4], 1:n_2, 1], traj[t_range[4], 1:n_2, 2], traj[t_range[4], 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
		Plots.scatter!(traj[t_range[4], (n_2+1):end, 1], traj[t_range[4], (n_2+1):end, 2],
		 traj[t_range[4], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"\mathcal{O}(t_{eddy})")


	if n_snaps==5
	p6 =Plots.scatter(traj[t_range[5], 1:n_2, 1], traj[t_range[5], 1:n_2, 2], traj[t_range[5], 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
		Plots.scatter!(traj[t_range[5], (n_2+1):end, 1], traj[t_range[5], (n_2+1):end, 2],
		traj[t_range[5], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_{eddy}")

		plt = plot(p1, p2, p3, p4, p5, p6, layout = (1, 6), legend = false)
	end
	if n_snaps==4
		plt = plot(p0, p1, p2, p3, p4, p5, layout = (1, 6), legend = false)
	end
	# title!("Learned WCSPH_$(method): N=$(N)")
	out_path = "./learned_figures/gen_t_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
	out_path1 = "./learned_figures/gen_1_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
	out_path2 = "./learned_figures/gen_2_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
	out_path3 = "./learned_figures/gen_3_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
	out_path4 = "./learned_figures/gen_4_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
	out_path5 = "./learned_figures/gen_5_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
	display(p1); display(p2); display(p3); display(p4); display(p5)
	# display(plt)
	# savefig(plt, out_path);
	savefig(p1, out_path1); savefig(p2, out_path2); savefig(p3, out_path3); savefig(p4, out_path4); savefig(p5, out_path5);
end



function obtain_snapshot(traj, t)
	m_s = 3.4
	gr(size=(600,500))
	n_2 = round(Int,N/2);

	p1 =Plots.scatter(traj[t, 1:n_2, 1], traj[t, 1:n_2, 2], traj[t, 1:n_2, 3],
		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s,
		color = "blue")
		# zlabel=L"\textrm{%$latex_method}", ztickfontsize=1, zguidefontsize=22)
		# zlabel=L"d_t v_i = NN_{\theta}", ztickfontsize=1, zguidefontsize=30)
		Plots.scatter!(traj[t, (n_2+1):end, 1], traj[t, (n_2+1):end, 2],
		traj[t,(n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_0")

	# title!("Learned WCSPH_$(method): N=$(N)")
	out_path = "./learned_figures/gen_t_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method)_$(t).png"
	display(p1);
	savefig(p1, out_path);
end


#
# if method!="truth"
# 	obtain_snapshots2(traj);
# end
#
#

if method=="truth"
	traj_gt_path = "./learned_data/traj_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
	vels_gt_path = "./learned_data/vels_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
	rhos_gt_path = "./learned_data/rhos_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"

	traj_gt = npzread(traj_gt_path)
	D = size(traj_gt)[3];
	N = size(traj_gt)[2];
	# obtain_snapshots2(traj_gt);
	# obtain_snapshot(traj_gt, 200)
end






#------- vary T, θ = 0.8e-1; (θ from training set)

traj_gt_path = "./learned_data/traj_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
vels_gt_path = "./learned_data/vels_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
rhos_gt_path = "./learned_data/rhos_N4096_T$(T)_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_truth.npy"
traj_gt = npzread(traj_gt_path);
vels_gt = npzread(vels_gt_path);
rhos_gt = npzread(rhos_gt_path);

N = size(traj_gt)[2];
D = size(traj_gt)[3];
m = (2. * pi)^D / N; #so that ρ₀ = 1;

include("./kde_G_3d.jl")
include("./loss_functions.jl")

θ_test = θ;
α_test = 1.0;
β_test = 2.0;
c_test = 10.0;

# println("****************** Obtaining interpolated fields ***********************")
# Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, T)
# Vf_pr,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj, vels, rhos, T)


function obtain_vf_t(t)
	Vf = zeros(t+1,N,D);
	Vf_truth = Vf_gt[1:(t+1), :, :];
	Vf_pred = Vf_pr[1:(t+1), :, :];
	return Vf_truth, Vf_pred
end


function obtain_gen_loss_t(t_s, t_end, t_num)
	t_range = ceil.(Int, range(t_s, t_end, length=t_num));
	num_t_idx = size(t_range)[1];
	println("num_t_idx  = ", num_t_idx);
	Lt = zeros(num_t_idx); Lgt = zeros(num_t_idx); rot_errt = zeros(num_t_idx);
	ii = 1;
	for t in t_range
		# Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, t)
		Diff_gt, Vel_inc_gt = obtain_gt_dists(traj_gt, vels_gt, t)
		Diff_pred, Vel_inc_pred = obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t);

		Vf_truth, Vf_pred = obtain_vf_t(t)
		Lt[ii] = compute_L_comp(loss_method, Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t)
		Lgt[ii] = compute_Lg(lg_method, Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t);

		println("T = ", t, "   L = ", Lt[ii], "   Lg = ", Lgt[ii]);
		ii += 1;
	end
	return Lt, Lgt
end


# Lt, Lgt = obtain_gen_loss_t(10, T, 20) #coarse = 1;

# make_dir("learned_generalization");
#
# save_output_data(Lt, "./learned_generalization/loss_t_$(method)_loss$(loss_method)_$(lg_method)_T$(T).npy")
# save_output_data(Lgt, "./learned_generalization/loss_g_t_$(method)_loss$(loss_method)_$(lg_method)_T$(T).npy")
#










#-------
""" GENERALIZATION OVER θ """

#-------


if method =="node"
	height = 5;
	# nn_data_dir = "./learned_models/output_data_node_kl_Vrandn_itr2000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint150_ts20_coarse1/"
		#doesnt learn advection properly with just kl loss!
	nn_data_dir = "./learned_models/output_data_forward_node_kl_lf_Vrandn_itr801_lr0.005_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch0/"
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
	p = params(NN); n_list = floor(Int, size(p[1])[2]/(2*D))
end

if method =="nnsum"
	height = 5;
	nn_data_dir = "./learned_models/output_data_forward_nnsum_kl_lf_Vrandn_itr600_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch0/"
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="rot_inv"
	height = 5;
	nn_data_dir = "./learned_models/output_data_forward_rot_inv_kl_lf_Vrandn_itr2000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch1/"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="eos_nn"
	height = 9;
	nn_data_dir = "./learned_models/output_data_forward_eos_nn_lf_Vrandn_itr2000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height9_klswitch1/"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="grad_p"
	height = 5;
	nn_data_dir = "./learned_models/output_data_forward_grad_p_kl_lf_Vrandn_itr500_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch0/"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="phys_inf"
	phys_data_dir = "./learned_models/output_data_forward_phys_inf_lf_Vrandn_itr3000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch1/"
	params_path = "$(phys_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
end



function include_sensitivity_file(method)
	include("./sensitivities_3d_$(method).jl")
end

include_sensitivity_file(method)

#
# if method!="truth"
# 	include_sensitivity_file(method)
# end
# if method=="truth"
# 	include_sensitivity_file("phys_inf")
# end
# include("./sph_3d_integrator.jl")
#
#
# # Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, 10)
#
# function obtain_gen_errs_over_θ(t)
# 	θ_range = 1e-1 : 0.3 : 1.7
# 	L = zeros(size(θ_range)[1]); Lg = zeros(size(θ_range)[1]);
# 	ii = 1;
# 	for θ_test in θ_range
# 		traj_test_set = npzread("./data/traj_N4096_T20_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ$(θ_test)_AV_neg_rel.npy");
# 		vels_test_set = npzread("./data/vels_N4096_T20_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ$(θ_test)_AV_neg_rel.npy");
# 		rhos_test_set = npzread("./data/rhos_N4096_T20_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ$(θ_test)_AV_neg_rel.npy");
#
# 		Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_test_set, vels_test_set, rhos_test_set, t)
#
# 		Diff_test, Vel_inc_test =
# 		obtain_pred_dists(traj_test_set[1:(t+1),:,:], vels_test_set[1:(t+1),:,:], traj_test_set[1,:,:], vels_test_set[1,:,:], t);
#
# 		traj_pred, vels_pred, rhos_pred =
# 		vel_verlet(traj_test_set, vels_test_set, p_fin, θ_test, t);
# 		Vf_pr,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, t)
#
# 		Diff_pred, Vel_inc_pred =
# 		obtain_pred_dists(traj_pred, vels_pred, traj_test_set[1,:,:], vels_test_set[1,:,:], t);
#
# 		L[ii] = compute_L_comp(loss_method, Vel_inc_test, Vel_inc_pred, traj_pred, vels_pred, rhos_pred, Vf_pr, Vf_gt, t)
# 		Lg[ii] = compute_Lg(lg_method, Vel_inc_test, Vel_inc_pred, traj_pred, vels_pred, rhos_pred, Vf_pr, Vf_gt, t);
#
# 		println("L = ", L[ii], "  Lg = ", Lg[ii]);
# 		ii += 1;
# 	end
# 	return L, Lg
# end


# Lθ, Lgθ = obtain_gen_errs_over_θ(10)
#
# save_output_data(Lθ, "./learned_generalization/Lθ_$(method)_$(loss_method)_$(lg_method).npy")
# save_output_data(Lgθ, "./learned_generalization/Lgθ_$(method)_$(loss_method)_$(lg_method).npy")


#
#
#
#
#
#
# #------ Generate figures
# # learned dists, translational, rotational, EoS,
#
#
# function load_data_files(pos_path, vel_path, rho_path)
#     println(" ****************** Loading $(latex_method) data files ***********************")
# 	traj = npzread(pos_path)
# 	vels = npzread(vel_path)
# 	rhos = npzread(rho_path)
# 	return traj, vels, rhos
# end
#
# if method!="truth"
# 	pos_path = "./learned_data/traj_N4096_T110_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_$(method).npy"
# 	vel_path = "./learned_data/vels_N4096_T110_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_$(method).npy"
# 	rho_path = "./learned_data/rhos_N4096_T110_ts1_h0.335_Vrandn_c10.0_α1.0_β2.0_θ0.5_$(method).npy"
# 	traj, vels, rhos = load_data_files(pos_path, vel_path, rho_path)
# 	D = size(traj)[3];
# 	N = size(traj)[2];
# end
#
# traj_pred, vels_pred, rhos_pred = load_data_files(pos_path, vel_path, rho_path);
# #
# # traj_pred, vels_pred, rhos_pred =
# # vel_verlet(traj_gt, vels_gt, p_fin, θ, 10);
#

# t_inc = 120;
# Diff_pred, Vel_inc_pred =
# obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t_inc);
# save_output_data(Vel_inc_pred, "./learned_generalization/vel_inc_pred_$(method)_loss$(loss_method)_t$(t_inc).npy")
# save_output_data(Diff_pred, "./learned_generalization/diff_pred_$(method)_loss$(loss_method)_t$(t_inc).npy")

# println("****************** Vel Inc Data Saved ****************")

#
# function comparing_Gu(G_u, Vel_inc_pred, θ, T, width=1.2)
#     gr(size=(500,500))
#     x_s = -width
#     x_e = width
#     G_pred(x) = kde(x, Vel_inc_pred[T, :, 1])
#     plt = plot(x->G_u(x), x_s, x_e, label=L"G(\tau, d)",
#                 color="indigo", linewidth = 4.5, legendfontsize=14)
#     plot!(x->G_pred(x), x_s, x_e, marker=:x, markersize=7, color="forestgreen",
#           markercolor = :black, label=L"\hat{G}_{\theta}(\tau, d)",
#           linestyle=:dash, linewidth = 4.5, legendfontsize=14)
#
#     title!(L"%$(latex_method)", titlefont=20)
#
#     # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
#     xlabel!(L"\textrm{Velocity - increment}", xtickfontsize=14, xguidefontsize=20)
#     ylabel!(L"G_{\delta u}", ytickfontsize=14, yguidefontsize=20)
#
#     display(plt)
#     out_path = "./learned_figures/G_trained_$(loss_method)_$(method)_kde_D$(D)_N$(N)_θ$(θ).png"
#
#     savefig(plt, out_path)
# end
#
# comparing_Gu(G_u, Vel_inc_pred, θ, T)
#
#




#--- translational and rotational

# function rotational_metric(X, V, p, θ, model_A)
#         Q,R = qr(randn(D,D)); Q = Q*Diagonal(sign.(diag(R))); #random orthogonal matrix
#         R_90 = [1.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 1.0 0.0];
# 		#R_90 rotation about x-axis
#         # Fqy, rh_ = model_A((Q*X')', (Q*V')', p)
#         Fry, rh_ = model_A((R_90*X')', (R_90*V')', θ, p)
#         F, rh_ = model_A(X, V, θ, p)
#         # QF = (Q*F')'
#         RF = (R_90 * F')'
#         return mse(Fry, RF)
# end
#
# shiftx = 2*pi*rand(); shiftv = 2*pi*rand();
# function translational_metric(X, V, p, θ, model_A)
#     # Fy_s, rh_ = model_A(X .- shiftx, V .- shiftv, p)
#     Fx_s, rh_ = model_A(X .- shiftx, V, θ, p)
#     F, rh_ = model_A(X, V, θ, p)
#     return mse(Fx_s, F)
# end


# rot_RF  = rotational_metric(traj_gt[1,:,:], vels_gt[1,:,:], p_fin, θ, obtain_sph_AV_A)
# gal_inv = translational_metric(traj_gt[1,:,:], vels_gt[1,:,:], p_fin, θ, obtain_sph_AV_A)
#
# save_output_data(rot_RF, "./learned_generalization/rot_RF_$(method)_loss$(loss_method).npy")
# save_output_data(gal_inv, "./learned_generalization/gal_inv_$(method)_loss$(loss_method).npy")
#
# println(rot_RF, ", ", gal_inv)

#
#
#
#
# #---- Plotting learned EoS
#
# function Pres(rho, c, g)
#   return c^2 * (rho^g - 1.) / g ;
# end
#


if method =="eos_nn"
	height = 10;
	nn_data_dir = "./learned_models/output_data_forward_eos_nn_lf_Vrandn_itr1200_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height10_klswitch0"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

function compare_eos(p_h)
    gr(size=(600,600))
    Pnn_comp(ρ) = re(p_h)([ρ])[1]
	max_rho = maximum(rhos_gt[1:11,:]);
	min_rho = minimum(rhos_gt[1:11,:]);
	mean_rho = mean(rhos_gt[1:11,:]);
	std_3 = 3*std(rhos_gt[1:11,:]);
	rho_std_p = mean_rho + std_3;
	rho_std_m = mean_rho - std_3;
	r_s = min_rho - 0.042;
	r_e = max_rho + 0.042;

	plt = plot(x -> Pres(x, c_gt, g), r_s, r_e, label=L"P_{truth}", color="blue", linewidth = 2.5)
    plot!(x -> Pnn_comp(x), r_s, r_e, marker=:x, markersize=5,
                  color="forestgreen", markercolor = :black,
                  label=L"P_{nn(\theta)}(\rho)", linestyle=:dash, linewidth = 2.5)

	vline!([min_rho, max_rho], linewidth = 1.5, color="orangered4", linestyle=:dash, label=L"(\rho_{min}, \rho_{max})")
	vline!([rho_std_p, rho_std_m], linewidth = 2.5, color="black", label=L"(\mu_{\rho} - 3\sigma_{\rho}, \mu_{\rho} + 3\sigma_{\rho})", linestyle=:dot)
	vline!([mean_rho], linewidth = 2.5, color="purple", linestyle=:dashdot, label=L"\rho_{avg}")

    title!(L"\textrm{Learning EoS with } L_f", titlefont=22)
    xlabel!(L"\rho", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"P(\rho)", ytickfontsize=12, yguidefontsize=20)

   display(plt)
   savefig(plt, "./learned_figures/EOS_$(loss_method)_$(method).png")
end

compare_eos(p_fin)
