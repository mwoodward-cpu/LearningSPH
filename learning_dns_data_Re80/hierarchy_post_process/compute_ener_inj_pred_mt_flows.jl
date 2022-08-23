"""
Using Trained models to make long time predcitions (9 times longer than training)

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra


class="phys"
m_num = 1;

# class="nns"
# m_num_nn = 3;

#-----Select mach number
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

include("./load_models_t20.jl")
T_train = 20; l_method = "lf"
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()
phys_dirs = [d1, d2, d3, d4];
nn_dirs = [d5, d6, d7, d8, d9];

if class=="phys"
	pdir = phys_dirs[m_num];
	method = m_phys[m_num];
end

if class=="nns"
	pdir = nn_dirs[m_num_nn];
	method = m_nns[m_num_nn];
end

l_method = "lf";
T = 20;
lg_method = "kl_lf";
T_pred = 70;

loss_method = l_method;
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 1.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 200;

extern_f = "determistic"
IC = "dns_equil"

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

if pdir in phys_dirs
	p_fin = load_learned_model_params(pdir)
end

if pdir in nn_dirs
	p_fin, NN, re = load_nn_learned_model(pdir)
	height = 6;
	if m_num_nn==1 || m_num_nn==4
		height = 8
	end
	n_params = size(p_fin)[1];
end

#to first compute just the learned acceleration term.
p_fin[end] = 0.0;

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

function comparing_Ga(acc_pred, T_frame, width=0.22)
    gr(size=(700,600))
	out_path = "./learned_figures_mt/ga_t_frame_$(method)_Tp$(T_pred).png"
    x_s = -width
    x_e = width
	Ga_m(x) = kde(x, vec(acc_pred[T_frame, :, :]));
	Ga_gt(x) = kde(x, vec(accl_gt[T_frame, :, :]));
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
    savefig(plt, out_path)
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

# comparing_Ga(accl, 20:T_pred, win)
comparing_Ga(accl, 2:T_pred, win)



function gen_data_files(accl, traj, vels, rhos)
	acc_path = "./learned_data/accl_theta_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
	pos_path = "./learned_data/traj_theta_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
	vel_path = "./learned_data/vels_theta_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
	rho_path = "./learned_data/rhos_theta_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"

    println(" ****************** Saving data files ***********************")
	# npzwrite(vor_path, vort[t_save:end,:,:])
	npzwrite(acc_path, accl)
	npzwrite(pos_path, traj)
	npzwrite(vel_path, vels)
	npzwrite(rho_path, rhos)
end

# gen_data_files(accl, traj, vels, rhos)
