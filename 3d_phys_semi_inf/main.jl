
"""
					Mixed Mode AD + Senstivity analysis SPH_AV

					KL divergence using only velocity increment δvᵢ


"""


using NPZ, Plots, Flux, QuadGK
using ForwardDiff
using Flux.Optimise: update!
using Flux.Optimise: Optimiser
using Flux.Losses
using BSON: @save


const T = 10;				#number of time steps in integration (forward prediction)
const coarse_mult = 1;		#coarse graining in time (set to 1)
const n_itrs = 3000;		#number of iteration
const vis_rate = 1;		#sampling frequency for output
const lr = 5e-2; 			#initial lr (later adapted with ADAM)
const t0 = 1; 				#time shift of data learned on
const mag = 1.0;			#Amplitude of external forcing
const h_kde = 0.9;			#factor of h for kde: silvermans rule 0.9
const r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
# nb = 1024;			#number of batches (if nb == N, this does relabeling of particles)
const nb = "all"			#don't permute particles and no batching
const n_int = 200; 		#number of integration points in KL (n_int >190 is sufficient)
const t_start = 20;             #time shift of IC of truth data to learn on.
const t_decay = round(Int, 0.9*n_itrs);			#time decay begins
const window = 1.2;	   				#domain for plotting G_u distributions (velocity increment dist'ns)
const height = 5;                                       #height of NN (see sensitivities of NN structure)

# sensitivity method:
sens_method = "forward"
# sens_method = "adjoint";

# loss_method = "kl_t"
# loss_method = "kl"
# loss_method = "kl_one_dist"
# loss_method = "lf"
# loss_method = "l2"
loss_method = "kl_lf"

switch_kl_lf = 1;  #switching to kl_lf loss function after 0.6*n_itrs
# switch_kl_lf = 0;   #keep same loss_method

# method = "node";
method = "nnsum"
# method = "rot_inv"
# method = "eos_nn"
# method = "grad_p"
# method = "phys_inf"

#-----physical parameters (known gt and initial guess)

const c = 10.0;
const h = 0.335;
const α = 1.0;
const β = 2.0;
const g = 7.0;
const θ = 5e-1;

#initial guess of physical paramters:
if method=="phys_inf"
	c_hat = rand(); α_hat = rand(); β_hat = rand(); g_hat = rand()
	p_hat = [c_hat, α_hat, β_hat, g_hat];
	n_params = size(p_hat)[1]
end


const c_gt = c;
const cdt = 0.4;
const dt = coarse_mult * cdt * h / c;

IC = "Vrandn"
# IC = "s_hit" #stationary homogeneous isotropic turbulence


println("switch_kl_lf = ", switch_kl_lf);
println(sens_method, "  ", loss_method, "  ", method)
println("coarse_mult = ", coarse_mult)


#-----Load data

const traj_gt = npzread("./data/traj_N4096_T50_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ0.5_AV_neg_rel.npy")[t_start:coarse_mult:end, :, :];
const vels_gt = npzread("./data/vels_N4096_T50_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ0.5_AV_neg_rel.npy")[t_start:coarse_mult:end, :, :];
const rhos_gt = npzread("./data/rhos_N4096_T50_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ0.5_AV_neg_rel.npy")[t_start:coarse_mult:end, :];


const N = size(traj_gt)[2];
const D = size(traj_gt)[3];
const m = (2. * pi)^D / N; #so that ρ₀ = 1;

#---Load utilities and function

function include_sensitivity_file(method)
	include("./sensitivities_3d_$(method).jl")
end

include_sensitivity_file(method)

include("./kde_G.jl")
include("./gen_outputs.jl")
include("./loss_functions.jl")
include("./integrators_utils.jl")

# plotting_Gvel_kde(window)
println("n_params = ", n_params)





#-------------- Mixed mode learning

#obtain ground truth velocity field
Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, T)


# opt = ADAM(lr);
function training_algorithm(l_method, n_itrs, vis_rate, T, p_h)
	loss_method = l_method;
	L_itr = zeros(round(Int, n_itrs/vis_rate))
	rot_RF = zeros(round(Int, n_itrs/vis_rate))
	gal_inv = zeros(round(Int, n_itrs/vis_rate))
	Vel_inc_pred_k = zeros(n_itrs, T, N);
	c_itr = zeros(round(Int, n_itrs/vis_rate))
	α_itr = zeros(round(Int, n_itrs/vis_rate))
	β_itr = zeros(round(Int, n_itrs/vis_rate))
	g_itr = zeros(round(Int, n_itrs/vis_rate))
	rho_data = 0.9:0.005:1.1; r_data = 0.0:0.01:2*h;
	P_nn = zeros(round(Int, n_itrs/vis_rate), size(rho_data)[1]);
	P_gt = Pres.(rho_data, c_gt, g);
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
		Vel_inc_pred_k[k, :, :] = Vel_inc_pred[:, :, 1];
        	if mod(k, vis_rate) == 0
			if method == "eos_nn"
				Pnn_comp(ρ) = re(p_h)([ρ])[1];
				P_nn[k, :] = Pnn_comp.(rho_data);
				compare_eos(p_h);
			end
			L = compute_L(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt,
				 			  traj_gt, traj_pred, vels_pred, rhos_pred)
			loss_method = kl_lf_switch(k, switch_kl_lf, loss_method, L);
			# if abs(L) < 5e-15
			#    opt = Optimiser(ExpDecay(lr, 0.1, 500, 1e-4), ADAM(lr))
			# end
			L_itr[ii] = L;
			if (method == "phys_inf")
				c_itr[ii] = p_h[1]; α_itr[ii] = p_h[2];
				β_itr[ii] = p_h[3]; g_itr[ii] = p_h[4];
				println("Itr  = ", k, "  c_hat = ", p_h[1],  "  α_hat = ", p_h[2], "  β_hat = ", p_h[3],
 						"  g_hat = ", p_h[4], "    Loss = ", L)
			end
			rot_Q, rot_RF[ii] = rotational_metric(traj_gt[1,:,:], vels_gt[1,:,:], p_h, obtain_sph_AV_A)
			gal_Y_shift, gal_inv[ii] = translational_metric(traj_gt[1,:,:], vels_gt[1,:,:], p_h, obtain_sph_AV_A)
			comparing_Gu(G_u, Vel_inc_pred, "train", θ, window)
			println("Itr  = ", k, "    $(loss_method): Loss = ", L)
			println("F(X-s) = ", gal_inv[ii], "   F(Y-s) = ", gal_Y_shift)
			save_output_data(p_h, "./$(data_out_path)/params_intermediate.npy")
			ii +=1;
        	end
		update!(opt, p_h, ∇L)
    end
    return p_h, Vel_inc_pred_k, P_nn, L_itr, rot_RF, gal_inv, c_itr, α_itr, β_itr, g_itr
end

p_fin, Vel_inc_pred_k, P_nn, L_out, rot_RF, galilean_inv, c_out, α_out, β_out, g_out =
	training_algorithm(loss_method, n_itrs, vis_rate, T, p_hat);


save_output_data(L_out, "./$(data_out_path)/loss.npy")
save_output_data(p_fin, "./$(data_out_path)/params_fin.npy")
save_output_data(rot_RF, "./$(data_out_path)/rot_error_rf.npy")
save_output_data(galilean_inv, "./$(data_out_path)/galilean_inv.npy")


plot_loss_itr()
plot_rot_itr()
plot_galilean_itr()
animate_Gu_fixt(n_itrs, Vel_inc_pred_k, window, 40)



if method == "eos_nn"
	rho_data = 0.9:0.005:1.1; P_gt = Pres.(rho_data, c_gt, g);
	animate_learning_EoS(n_itrs, rho_data, P_gt, P_nn, 40)
end

if method == "phys_inf"
	plot_4g_param()
	save_output_data(c_out, "./$(data_out_path)/c_out.npy")
	save_output_data(α_out, "./$(data_out_path)/alpha_out.npy")
	save_output_data(β_out, "./$(data_out_path)/beta_out.npy")
	save_output_data(g_out, "./$(data_out_path)/g_out.npy")
end
