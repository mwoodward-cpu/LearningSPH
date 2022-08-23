"""
Using Trained models to make long time predcitions (9 times longer than training)

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra


# Mt = 0.04
# mmt = "004";

# Mt = 0.08
# mmt = "008";

Mt = 0.16
mmt = "016";

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];


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

T_train = 20; l_method = "lf"
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()
phys_dirs = [d1, d2, d3, d4];
nn_dirs = [d5, d6, d7, d8, d9];

m_num = 1;
pdir = phys_dirs[m_num];
method = m_phys[m_num];

l_method = "lf";
T = 20;
lg_method = "kl_lf";
T_pred = 250;

loss_method = l_method;
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 1.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 200;

extern_f = "determistic"
IC = "dns_equil"

θ = 0.0002;
h = 0.335;
t_coarse = 1
dt = t_coarse*0.04;


include("./data_loader.jl")
pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

x0 = traj_gt[1, :, :]
v0 = vels_gt[1, :, :]
rho0 = rhos_gt[1, :]


D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


println("*****************    *************")
println("running method = ", method)
include("load_models_t20.jl")
p_fin = load_learned_model_params(pdir)

println(p_fin)
p_fin[end] = 0.0; #to compute just the learned acceleration term.
println(p_fin)


function include_sensitivity_file(method)
	include("./models/sensitivities_3d_$(method).jl")
end

if method!="truth"
	include_sensitivity_file(method)
	A, rho_ = obtain_sph_AV_A(x0, v0, p_fin)
end

#compute_v_dot_a:
function compute_v_dot_a(v,a)
	v_dot_a = zeros(N);
	for i in 1 : N
		v_dot_a[i] = v[i,:]'*a[i,:];
	end
	return v_dot_a
end

v_dot_a = compute_v_dot_a(v0,A) #computes <v⋅F>
v_F_mean = mean(v_dot_a)

#compute KE
ke = 0.5/N * sum(rho0.*(v0[:,1].^2 .+ v0[:,2].^2 .+ v0[:,3].^2));
v_dot_v = compute_v_dot_a(v0,v0);

v2_ke_mean = mean(v_dot_v ./ ke) #computes <v^2/ke>

θ_inj = -v_F_mean / v2_ke_mean

println("Energy injection rate for method ", method)
println(" is = ", θ_inj)


#Make forward predictions with new energy injection rate
println("==================================")
println("Forward predictions with θ_inj")
p_fin[end] = θ_inj
# println(p_fin)

if method!="truth"
	include_sensitivity_file(method)
	include("./sph_3d_integrator.jl")
	accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, T_pred)
end






#---------- Now make predictions with this energy rate and check acc stats


#check accl stats:
function compute_gt_accl(vel_gt)
	accl_gt = zeros(T_pred, N, D);
	for t in 1 : (T_pred-1)
		for i in 1 : D
			accl_gt[t+1,:,i] = (vel_gt[t+1,:,i] .- vel_gt[t,:,i]) / dt
		end
	end
	return accl_gt
end

h_kde = 0.9
include("kde_G.jl")
accl_gt = compute_gt_accl(vels_gt)

function comparing_Ga(acc_pred, T_scale, width=0.22)
    gr(size=(700,600))
    x_s = -width
    x_e = width
	Ga_m(x) = kde(x, vec(acc_pred[T_scale, :, :]));
	Ga_gt(x) = kde(x, vec(accl_gt[T_scale, :, :]));
	# [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
	plt = plot(x->Ga_m(x), x_s, x_e, label="pred",
          linestyle=:dash, linewidth = 2.5, legendfontsize=10, color="maroon")
	plot!(x->Ga_gt(x), x_s, x_e, label=L"G_{truth}",
		  linewidth = 2.5, legendfontsize=10, color="indigo")


	title!("$(method), Mt = $(Mt)", titlefont=20)
    # title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Acceleration} ", xtickfontsize=14, xguidefontsize=20)
    ylabel!(L"G_a", ytickfontsize=14, yguidefontsize=20)

    display(plt)
    # out_path = "./learned_figures/Ga_trained_comp_Tt$(T_train)_Mt$(mmt).png"
    # savefig(plt, out_path)
end

if Mt==0.04
	win = 0.03
end

if Mt==0.08
	win = 0.2
end

if Mt==0.16
	win = 0.4
end

comparing_Ga(accl, T_pred, win)
