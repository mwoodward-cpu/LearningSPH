
"""
					Mixed Mode AD + Senstivity analysis SPH_AV

				      Learing hierarhcy of parameterized SPH models


"""


using NPZ, Plots, Flux, QuadGK
using ForwardDiff
using Flux.Optimise: update!
using Flux.Optimise: Optimiser
using Flux.Losses
using BSON: @save


const t_coarse = 2
const vis_rate = 1;		#sampling frequency for output
const lr = 2e-2; 			#initial lr (later adapted with ADAM)
const t0 = 1; 				#time shift of data learned on
const mag = 1.0;			#Amplitude of external forcing
const h_kde = 0.9;			#factor of h for kde: silvermans rule 0.9
const r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
const n_int = 220; 		#number of integration points in KL (n_int >190 is sufficient)
const t_start = 1;             #time shift of IC of truth data to learn on.
const height = 6;                                       #height of NN (see sensitivities of NN structure)

# loss_method = ARGS[1];
# sens_method = ARGS[2];
# switch_kl_lf = ARGS[3];
# ic_method = ARGS[4];
# const T = parse(Int, ARGS[5]);
# const n_itrs = parse(Int, ARGS[6]);		#number of iteration
# const t_decay = round(Int, 0.9*n_itrs);			#time decay begins

loss_method = "lf";
sens_method = "forward";
switch_kl_lf = 0;
ic_method = "unif_tracers";
T = 22;
n_itrs = 100;		#number of iteration
t_decay = round(Int, 0.9*n_itrs)
window = T*0.07/10;
s_itr = 0.5 * n_itrs


println("loss_method = ", loss_method)
println("sens_method = ", sens_method)
println("switch_method = ", switch_kl_lf)
println("ic_method = ", ic_method)

method = "phys_inf_theta_po_liv_Pi"
extern_f = "determistic"
# extern_f = "none"

#-----physical parameters (known gt and initial guess)

c = 0.845;
c_gt = c;
# h = 0.4;
h = 0.335
α = 0.2;
β = 0.4;
g = 1.4;
θ_gt = 0.0012;
θ = θ_gt;
println(" theta = ", θ)


c_hat = c; α_hat = α; β_hat = β; g_hat = g;
θ_hat = θ; po_hat = 0.0;
p_hat = [c_hat, α_hat, β_hat, g_hat, po_hat, θ_hat];
n_params = size(p_hat)[1]


println("T = ", T, "  n_itr = ", n_itrs)
println("extern_f = ", extern_f)



#-----Load Tracers Data
include("../data_loader.jl")

pos_path = "../wc_dns_unif_4096_gen_data_mt0.08/pos_traj_4k.npy"
vel_path = "../wc_dns_unif_4096_gen_data_mt0.08/vel_traj_4k.npy"
rho_path = "../wc_dns_unif_4096_gen_data_mt0.08/rho_traj_4k.npy"


traj_, vels_, rhos_ = load_dns_tracers(pos_path, vel_path, rho_path)
garb_, N, D = size(traj_);

traj_ = periodicize(traj_, N)
dt = t_coarse * 2e-2;
traj_gt, vels_gt, rhos_gt = coarse_grain_time(traj_, vels_, rhos_, t_coarse, N, D)

gsp = 4;
const N_f = size(traj_gt)[2];
const m = (2. * pi)^D / N; #so that ρ₀ = 1;


#---Load utilities functions and sensitivities

function include_sensitivity_file(method)
	include("../sensitivities/sensitivities_3d_$(method).jl")
end

include_sensitivity_file(method)
println("n_params = ", n_params)
include("../kde_G.jl")
include("../gen_outputs.jl")
include("../loss_functions.jl")
include("../integrators_utils.jl")


#-------------- Mixed mode learning


# opt = ADAM(lr);
function training_algorithm(l_method, n_itrs, vis_rate, T, p_h)
	loss_method = l_method; range_itrs = round(Int, n_itrs/vis_rate)
	L_itr = zeros(range_itrs)
	Vel_inc_pred_k = zeros(range_itrs, T, N);
	Vfp_k = zeros(range_itrs, T+1, N, D);
	c_itr = zeros(range_itrs); α_itr = zeros(range_itrs);
	β_itr = zeros(range_itrs); g_itr = zeros(range_itrs);
	a_itr = zeros(range_itrs); b_itr = zeros(range_itrs);
	θ_itr = zeros(range_itrs); po_itr = zeros(range_itrs);
	ii = 1;
	opt = Optimiser(ExpDecay(lr, 0.1, t_decay, 1e-4), ADAM(lr))
	for k ∈ 1 : n_itrs
		if sens_method=="forward"
			traj_pred, vels_pred, rhos_pred, ST = simultaneous_integration(p_h, T)
			Diff_pred, Vel_inc_pred = obtain_pred_dists(traj_pred, vels_pred, traj_gt[1,:,:], vels_gt[1,:,:])
			∇L = compute_∇L(loss_method, Vel_inc_gt, Vel_inc_pred, traj_pred, traj_gt, vels_pred, vels_gt, rhos_pred, Vf_gt, ST);
		end
		if sens_method=="adjoint"
			traj_pred, vels_pred, rhos_pred, λT, ∂F_pT = dual_adjoint_integration(p_h, T)
			Diff_pred, Vel_inc_pred = obtain_pred_dists(traj_pred, vels_pred, traj_gt[1,:,:], vels_gt[1,:,:])
			∇L = compute_adjoint_∇L(λT, ∂F_pT)
		end
        if mod(k, vis_rate) == 0
			Vfp, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w =
			obtain_interpolated_velocity_over_τ(X_grid, traj_pred, vels_pred, rhos_pred, T, N);

			Vfp_k[ii, :, :, :] = Vfp;
			Vel_inc_pred_k[ii, :, :] = Vel_inc_pred[:, :, 1];
			L = compute_L(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt,
				 			  traj_gt, traj_pred, vels_pred, rhos_pred, Vfp)
			loss_method = kl_lf_switch(k, switch_kl_lf, loss_method, L);

			L_itr[ii] = L;
			c_itr[ii] = p_h[1]; α_itr[ii] = p_h[2];
			β_itr[ii] = p_h[3]; g_itr[ii] = p_h[4]; po_itr[ii] = p_h[5]; θ_itr[ii] = p_h[6];
			println("Itr  = ", k, "  c_hat = ", p_h[1],  "  α_hat = ", p_h[2], "  β_hat = ", p_h[3],
			   	     "  g_hat = ", p_h[4], "  p_hat = ", p_h[5], "  θ_hat = ", p_h[6], "   $(loss_method): Loss = ", L)

			comparing_Gu(G_u, Vel_inc_pred, "train", θ, window);
			comparing_Ga(vels_pred, 0.22)
			plot_vel_contours(Vfp, Vf_gt, gsp, T) #Obtain Vfp each iter.
			save_output_data(p_h, "./$(data_out_path)/params_intermediate.npy")
			ii +=1;
        end
		update!(opt, p_h, ∇L)
    end
    return p_h, Vel_inc_pred_k, Vfp_k, L_itr, c_itr, α_itr, β_itr, g_itr, a_itr, b_itr, po_itr, θ_itr
end

p_fin, Vel_inc_pred_k, Vfp_k, L_out, c_out, α_out, β_out, g_out, a_out, b_out, po_out, θ_out =
	training_algorithm(loss_method, n_itrs, vis_rate, T, p_hat);


save_output_data(L_out, "./$(data_out_path)/loss.npy")
save_output_data(p_fin, "./$(data_out_path)/params_fin.npy")
animate_Gu_fixt(n_itrs, Vel_inc_pred_k, window, 40)
animate_vel_coutours(n_itrs, Vfp_k, Vf_gt, T+1, 40)

   save_output_data(c_out, "./$(data_out_path)/c_out.npy")
   save_output_data(α_out, "./$(data_out_path)/alpha_out.npy")
   save_output_data(β_out, "./$(data_out_path)/beta_out.npy")
   save_output_data(g_out, "./$(data_out_path)/g_out.npy")
   save_output_data(po_out, "./$(data_out_path)/po_out.npy")
   save_output_data(θ_out, "./$(data_out_path)/theta_out.npy")
   plot_6pθ_param()


plot_loss_itr()
