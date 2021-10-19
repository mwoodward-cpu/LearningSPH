

"""
2D fsa with forwardiff AD

learning less informed --> physics informed SPH
"""


using NPZ, Plots, Flux, QuadGK
using ForwardDiff, Statistics
using Flux.Optimise: update!
using Flux.Optimise: Optimiser
using Flux.Losses
#import Pkg; Pkg.add("BSON")
using BSON: @save


const T = 15				#number of time steps in integration (prediction step)
const coarse_mult = 1;  #coarse graining in time (number of dts to skip)
const n_itrs = 1000			#number of iteration
const vis_rate = 1;		#sampling frequency for output
const lr = 5e-2 			#initial lr (later adapted with ADAM)
const mag = 1.0			#Amplitude of external forcing
const r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
const h_kde = 0.9;	#smoothing parameter h; using simpsons rule;
const nb = "all";				# number of samples in batching
const n_int = 220;  	#number of integration points in numerical computation of KL
const t_start = 3205;    # time just after stationarity is reached
const window = 0.2;
const height = 8;
const t_decay = ceil(Int, 0.6*n_itrs);			#iteration decay of lr begins

println("height = ", height)


# sensitivity method:
sens_method = "forward"
# sens_method = "adjoint"

# loss_method = "l2"
# loss_method = "kl"
# loss_method = "lf"
# loss_method = "kl_t"
# loss_method = "kl_l2_t"
# loss_method = "kl_one_dist"
# loss_method = "kl_t_one_dist"
loss_method = "lf"
# loss_method = "kl_lf_t"
# loss_method = "kl_lf"

# switch_kl_lf = 0; s_itr = n_itrs;
switch_kl_lf = 1; s_itr = 0.5 * n_itrs;
# switch_kl_lf = 1; s_itr = 20;

# method = "node"
# method = "nnsum"
# method = "nnsum2"
# method = "rot_nn"
# method = "eos_nn"
# method = "grad_p"
method = "phys_inf"

# IC = "Vrandn"
IC = "s_hit"


println("switch_kl_lf = ", switch_kl_lf);
println(sens_method, " ", loss_method, " ", method)
println("coarse_mult = ", coarse_mult)

using Dates
println(Dates.now())


#----- Load data

const c_gt = 12.0;
const c = c_gt;
const h = 0.2;
const α = 1.0;
const β = 2.0;
const g = 7.0;
const θ = 8e-2;

const cdt = 0.4;
const dt = coarse_mult * cdt * h / c_gt;

const traj_gt = npzread("./data/traj_N1024_T6001_ts5801_h0.2_s_hit_cdt0.4_c12.0_α1.0_β2.0_θ0.08_AV_neg_rel_ke.npy")[t_start:coarse_mult:end, :, :];
const vels_gt = npzread("./data/vels_N1024_T6001_ts5801_h0.2_s_hit_cdt0.4_c12.0_α1.0_β2.0_θ0.08_AV_neg_rel_ke.npy")[t_start:coarse_mult:end, :, :];
const rhos_gt = npzread("./data/rhos_N1024_T6001_ts5801_h0.2_s_hit_cdt0.4_c12.0_α1.0_β2.0_θ0.08_AV_neg_rel_ke.npy")[t_start:coarse_mult:end, :];


const N = size(traj_gt)[2];
const D = size(traj_gt)[3];
const m = (2. * pi)^D / N; #so that ρ₀ = 1;


if method=="phys_inf"
	c_hat = rand(); α_hat = rand(); β_hat = rand(); g_hat = rand()
	p_hat = [c_hat, α_hat, β_hat, g_hat];
	n_params = size(p_hat)[1]
end


#--- Load files

function include_sensitivity_file(method)
	include("./sensitivities_$(method).jl")
end

include_sensitivity_file(method)

include("./gen_outputs_2d.jl")
include("./loss_functions_2d.jl")
include("./kde_G.jl")
include("./integrators_utils.jl")

println("n_params = ", n_params)



#------------------- Training algorithm

#obtaining velocity field of ground truth
Vf_gt, d1, d2, d3, d4 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, T)

function training_algorithm(l_method, n_itrs, vis_rate, T, p_h)
	loss_method = l_method;
	L_itr = zeros(round(Int, n_itrs/vis_rate))
	Vel_inc_pred_k = zeros(n_itrs, T, N);
	rot_QF = zeros(round(Int, n_itrs/vis_rate))
	rot_RF = zeros(round(Int, n_itrs/vis_rate))
	gal_inv = zeros(round(Int, n_itrs/vis_rate))
	Lg_itr = zeros(round(Int, n_itrs/vis_rate))
	c_itr = zeros(round(Int, n_itrs/vis_rate))
	α_itr = zeros(round(Int, n_itrs/vis_rate))
	β_itr = zeros(round(Int, n_itrs/vis_rate))
	g_itr = zeros(round(Int, n_itrs/vis_rate))
	rho_data = 0.9:0.005:1.1; r_data = 0.0:0.01:2*h;
	P_nn = zeros(round(Int, n_itrs/vis_rate), size(rho_data)[1]);
	P_gt = Pres.(rho_data, c_gt, g);
	ii = 1;
	# opt = ADAM(lr); #optimizer for gradient descent
	opt = Optimiser(ExpDecay(lr, 0.1, t_decay, 1e-4), ADAM(lr))
	for k ∈ 1 : n_itrs
		if sens_method=="forward"
        	traj_pred, vels_pred, rhos_pred, HT = simultaneous_integration(p_h, T)
			Diff_pred, Vel_inc_pred = obtain_pred_dists(traj_pred, vels_pred, traj_gt[1,:,:], vels_gt[1,:,:])
			∇L = compute_∇L(loss_method, Vel_inc_gt, Vel_inc_pred, traj_pred, traj_gt, vels_pred, vels_gt, rhos_pred, Vf_gt, HT);
		end
		if sens_method=="adjoint"
			traj_pred, vels_pred, rhos_pred, λT, ∂F_pT = dual_adjoint_integration(p_h, T)
			Diff_pred, Vel_inc_pred = obtain_pred_dists(traj_pred, vels_pred, traj_gt[1,:,:], vels_gt[1,:,:])
			∇L = compute_adjoint_∇L(λT, ∂F_pT)
		end
		update!(opt, p_h, ∇L)
		Vel_inc_pred_k[k, :, :] = Vel_inc_pred[:, :, 1];
        if mod(k, vis_rate) == 0
			if method == "eos_nn"
				Pnn_comp(ρ) = re(p_h)([ρ])[1];
				P_nn[k, :] = Pnn_comp.(rho_data);
				compare_eos(p_h);
			end
			rot_QF[ii], rot_RF[ii] = rotational_metric(traj_gt[1,:,:], vels_gt[1,:,:], p_h, obtain_sph_AV_A);
			L_itr[ii] = compute_L(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt,
							  traj_gt, traj_pred, vels_pred, rhos_pred, Vf_gt);

			loss_method = kl_lf_switch(k, switch_kl_lf, loss_method, s_itr, L_itr[ii]);
			gal_Y_shift, gal_inv[ii] = translational_metric(traj_gt[1,:,:], vels_gt[1,:,:], p_h, obtain_sph_AV_A)
			# if abs(L_itr[ii]) < 5e-3
			# 	opt = Optimiser(ExpDecay(lr, 0.1, 500, 1e-4), ADAM(lr))
			# end
			if (method == "phys_inf")
				c_itr[ii] = p_h[1]; α_itr[ii] = p_h[2];
				β_itr[ii] = p_h[3]; g_itr[ii] = p_h[4];
				println("Itr  = ", k, "  c_hat = ", p_h[1],  "  α_hat = ", p_h[2], "  β_hat = ", p_h[3],
						"  g_hat = ", p_h[4], "    Loss = ", L_itr[ii])
			end
			println("Itr  = ", k, " $(loss_method):  Loss = ", L_itr[ii])
			ii += 1;
			save_output_data(p_h, "./$(data_out_path)/params_intermediate.npy")
        end
		comparing_Gu(G_u, Vel_inc_pred, "train", θ, window)
    end
	# animate_learning(n_itrs, rho_data, P_gt, P_nn)
	return L_itr, rot_QF, rot_RF, gal_inv, Vel_inc_pred_k, p_h, P_nn, c_itr, α_itr, β_itr, g_itr
end


L_out, rot_QF, rot_RF, galilean_inv, Vel_inc_pred_k, p_fin, P_nn, 
c_out, α_out, β_out, g_out  =
	training_algorithm(loss_method, n_itrs, vis_rate, T, p_hat)

L_out = abs.(L_out)

println(Dates.now())

save_output_data(L_out, "./$(data_out_path)/loss.npy")
save_output_data(rot_QF, "./$(data_out_path)/rot_error_qf.npy")
save_output_data(rot_RF, "./$(data_out_path)/rot_error_rf.npy")
# save_output_data(Vel_inc_pred_k, "./$(data_out_path)/vel_inc_pred_k.npy")
save_output_data(p_fin, "./$(data_out_path)/params_final.npy")
save_output_data(galilean_inv, "./$(data_out_path)/galilean_inv.npy")


plot_loss_itr()
plot_rot_itr()
plot_galilean_itr()
animate_Gu_fixt(n_itrs, Vel_inc_pred_k, window, 40)

if (method != "phys_inf")
	@save "./$(data_out_path)/NN_model.bson" NN
end

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
