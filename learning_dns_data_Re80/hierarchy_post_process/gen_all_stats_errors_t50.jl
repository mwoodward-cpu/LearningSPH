using Plots, NPZ, LaTeXStrings, Flux
using Flux.Losses, Statistics

function save_output_data(data, path)
    npzwrite(path, data)
end
function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end


m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];

m_tot = vcat(m_phys, m_nns);
gen_error_m = "over_Mt"
# gen_error_m = "over_t"


# method = m_phys[1];
l_method = "lf_klt";
T = 50;
lg_method = "kl_lf";
T_train = 50; T_pred = 500;
T_loss = 100

loss_method = "lf";
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 250;

extern_f = "determistic"
IC = "dns_equil"


θ = 0.0002;
h = 0.335;
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


Vf_gt = npzread("./learned_field_data_t50/Vf_gt_Tp500.npy")

for method in m_tot[1:4]

		function compute_gt_accl(vel_gt)
			accl_gt = zeros(T_pred, N, D);
			for t in 1 : (T_pred-1)
				for i in 1 : D
					accl_gt[t+1,:,i] = (vel_gt[t+1,:,i] .- vel_gt[t,:,i]) / dt
				end
			end
			return accl_gt
		end

		accl_gt = compute_gt_accl(vels_gt)


		println("*****************    *************")
		println("Running Method = ", method)


		function load_data_t20(method, T_pred, T_train)
			data_dir_phy = "./learned_data_t50"
			acc_path = "$(data_dir_phy)/accl_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
			tra_path = "$(data_dir_phy)/traj_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
			vel_path = "$(data_dir_phy)/vels_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
			rho_path = "$(data_dir_phy)/rhos_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
			accl = npzread(acc_path);
			traj = npzread(tra_path);
			vels = npzread(vel_path);
			rhos = npzread(rho_path);
			return accl, traj, vels, rhos
		end

		if gen_error_m=="over_t"
			accl, traj, vels, rhos = load_data_t20(method, T_pred, T_train);
		end

		function load_vel_field_data(method)
			data_dir_phy = "./learned_fields_t50"
			vfpr_path = "$(data_dir_phy)/vel_field_$(method)_t240.npy"
			vfgt_path = "$(data_dir_phy)/vel_field_truth_t240.npy"
			vf_pr = npzread(vfpr_path);
			vf_gt = npzread(vfgt_path);
			return vf_pr, vf_gt
		end

		# Vf_pr, Vf_gt = load_vel_field_data(method);
		Vf_pr = npzread("./learned_field_data_t50/Vf_pr_$(method)_$(l_method)_Tp$(T_pred)_Tt$(T_train).npy")
		# Vf_pr = reshape(Vf_pr, (241, N, D))
		# Vf_gt = reshape(Vf_gt, (241, N, D))
		# include("./loss_functions.jl")


		"""
		==========================================================================
		Obtaining generalization error data: over time
		==========================================================================

		"""

		include("./kde_G.jl")
		include("./loss_functions_gen.jl")
		make_dir("learned_generalization_t50");

		function obtain_vf_t(t)
		        Vf = zeros(t+1,N,D);
		        Vf_truth = Vf_gt[1:(t+1), :, :];
		        Vf_pred = Vf_pr[1:(t+1), :, :];
		        return Vf_truth, Vf_pred
		end

		function compute_avg_kinetic_energy(rhos, vels, t)
		    tke = zeros(t);
		    for i in 1 : t
		        tke[i] = 0.5*mean(rhos[t,:] .* (vels[t, : ,1].^2 .+ vels[t, : ,2].^2 .+ vels[t, : ,3].^2));
		    end
		    avg_ke = mean(tke);
		    return avg_ke;
		end

		include("kl_loss_zdata.jl")

		function obtain_gen_loss_t(t_s, t_end, t_num)
		        t_range = ceil.(Int, range(t_s, t_end, length=t_num));
		        num_t_idx = size(t_range)[1];
		        println("num_t_idx  = ", num_t_idx);
		        Lt = zeros(num_t_idx); Lgt = zeros(num_t_idx); rot_errt = zeros(num_t_idx);
				Lgkl_t = zeros(num_t_idx); Lf_norm = zeros(num_t_idx);
		        Lkl_a = zeros(num_t_idx); Lkl_d = zeros(num_t_idx);
		        ii = 1;
		        for t in t_range
		                Diff_gt, Vel_inc_gt = obtain_gt_dists(traj_gt, vels_gt, t)
		                Diff_pred, Vel_inc_pred = obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t);
		                avg_ke = compute_avg_kinetic_energy(rhos_gt, vels_gt, t);
		                Vf_truth, Vf_pred = obtain_vf_t(t);
		                Lt[ii] = compute_L_comp(loss_method, Vel_inc_gt, Vel_inc_pred, Vf_pred, Vf_truth, t)
		                Lgt[ii] = compute_Lg(lg_method, Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t);
						Lgkl_t[ii] = Ikl_τ_zdat(Vel_inc_gt, Vel_inc_pred, t, 50);
		                Lf_norm[ii] = compute_Lg("lf", Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t)/avg_ke;
						Lkl_a[ii] = Ikl_τ_zdat(accl_gt, accl, t, 50);
						Lkl_d[ii] = Ikl_τ_diff(Diff_gt, Diff_pred, t, 50);
		                println("T = ", t, "   Loss: $(loss_method) = ", Lt[ii], "  $(lg_method) = ", Lgt[ii], "   kl_t = ", Lgkl_t[ii], "  Lf_norm = ", Lf_norm[ii],
								 "   kla_t = ", Lkl_a[ii], "  Lkl_d = ", Lkl_d[ii]);
		                ii += 1;
		        end
		        return Lt, Lgt, Lgkl_t, Lf_norm, Lkl_a, Lkl_d
		end


		if gen_error_m == "over_t"
			Lt, Lgt, Lgkl_t, Lf_norm, Lkl_a, Lkl_d = obtain_gen_loss_t(50, T_loss, 20)
			# save_output_data(Lt, "./learned_generalization_t50/$(l_method)_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train)_$(gen_error_m).npy")
			save_output_data(Lgt, "./learned_generalization_t50/$(lg_method)_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train)_$(gen_error_m).npy")
			save_output_data(Lgkl_t, "./learned_generalization_t50/kl_t_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train)_$(gen_error_m).npy")
			# save_output_data(Lf_norm, "./learned_generalization_t50/lf_norm_loss_t_$(method)_Tp$(T_pred)_Tt$(T_train)_$(gen_error_m).npy")
			save_output_data(Lkl_a, "./learned_generalization_t50/kl_t_accl_$(method)_Tp$(T_pred)_Tt$(T_train)_$(gen_error_m).npy")
			save_output_data(Lkl_d, "./learned_generalization_t50/kl_t_diff_$(method)_Tp$(T_pred)_Tt$(T_train)_$(gen_error_m).npy")
		end



		"""
		==========================================================================
		Obtaining generalization error data: over Mt
		==========================================================================

		"""


		function load_gen_mt_gt_data(mmt)
			pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
			vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
			rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
			traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)
			# Vf_gt = npzread("./learned_field_data/vf_gt_Mt$(mt).npy")
			return traj_gt, vels_gt, rhos_gt
		end

		function load_gen_mt_pr_data(method, Mt)
			data_dir_phy = "./learned_data_t50"
			acc_path = "$(data_dir_phy)/accl_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
			tra_path = "$(data_dir_phy)/traj_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
			vel_path = "$(data_dir_phy)/vels_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
			rho_path = "$(data_dir_phy)/rhos_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(Mt).npy"
			accl = npzread(acc_path);
			traj = npzread(tra_path);
			vels = npzread(vel_path);
			rhos = npzread(rho_path);
			return accl, traj, vels, rhos
		end

		include("./loss_functions.jl")

		function obtain_vel_mt_field_values(t, traj_in, vels_in, rhos_in)
			Vf,d1_,d2_,d3_,d4_,d5_,d6_ = obtain_interpolated_velocity_over_τ(X_grid, traj_in, vels_in, rhos_in, t, N_f)
			return Vf
		end

		function string_mt(mt)
			if mt==0.04 mmt = "004" end
			if mt==0.08 mmt = "008" end
			if mt==0.16 mmt = "016" end
			return mmt
		end

		function obtain_gen_loss_Mt(t, Mt)
		        num_mt_idx = size(Mt)[1];
		        println("num_mt_idx  = ", num_mt_idx);
		        Lt = zeros(num_mt_idx); Lgt = zeros(num_mt_idx);
				Lgkl_t = zeros(num_mt_idx); Lf_norm = zeros(num_mt_idx);
		        Lkl_a = zeros(num_mt_idx); Lkl_d = zeros(num_mt_idx);
		        ii = 1;
		        for mt in Mt
						mmt = string_mt(mt)
						traj_gt, vels_gt, rhos_gt = load_gen_mt_gt_data(mmt);
						Vf_truth = obtain_vel_mt_field_values(t, traj_gt, vels_gt, rhos_gt);
						accl, traj, vels, rhos = load_gen_mt_pr_data(method, mt)
						Vf_pred = obtain_vel_mt_field_values(t, traj, vels, rhos);
						println("size Vf = ", size(Vf_pred))
						# save_output_data(Vf_truth, "./learned_field_data/vf_gt_mt$(mt).npy")
						# save_output_data(Vf_pred, "./learned_field_data/vf_pr_$(method)_$(l_method)_mt$(mt).npy")

		                Diff_gt, Vel_inc_gt = obtain_gt_dists(traj_gt, vels_gt, t)
		                Diff_pred, Vel_inc_pred = obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t);
		                avg_ke = compute_avg_kinetic_energy(rhos_gt, vels_gt, t);
		                # Vf_truth, Vf_pred = obtain_vf_t(t)
		                Lt[ii] = compute_L_comp(loss_method, Vel_inc_gt, Vel_inc_pred, Vf_pred, Vf_truth, t)
		                Lgt[ii] = compute_Lg(lg_method, Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t);
						Lgkl_t[ii] = Ikl_τ_zdat(Vel_inc_gt, Vel_inc_pred, t, 16);
		                Lf_norm[ii] = compute_Lg("lf", Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t)/avg_ke;
						Lkl_a[ii] = Ikl_τ_zdat(accl_gt, accl, t, 16);
						Lkl_d[ii] = Ikl_τ_diff(Diff_gt, Diff_pred, t, 16);
		                println("T = ", t, "   Loss: $(loss_method) = ", Lt[ii], "  $(lg_method) = ", Lgt[ii], "   kl_t = ", Lgkl_t[ii], "  Lf_norm = ", Lf_norm[ii],
								 "   kla_t = ", Lkl_a[ii], "  Lkl_d = ", Lkl_d[ii]);
		                ii += 1;
		        end
		        return Lt, Lgt, Lgkl_t, Lf_norm, Lkl_a, Lkl_d
		end


		if gen_error_m == "over_Mt"
			Lt, Lgt, Lgkl_t, Lf_norm, Lkl_a, Lkl_d = obtain_gen_loss_Mt(T_train, [0.04, 0.08, 0.16])
			save_output_data(Lt, "./learned_generalization_t50/$(l_method)_loss_t_$(method)_T$(T_train)_$(gen_error_m).npy")
			save_output_data(Lgt, "./learned_generalization_t50/$(lg_method)_loss_t_$(method)_T$(T_train)_$(gen_error_m).npy")
			save_output_data(Lgkl_t, "./learned_generalization_t50/kl_t_loss_t_$(method)_T$(T_train)_$(gen_error_m).npy")
			save_output_data(Lf_norm, "./learned_generalization_t50/lf_norm_loss_t_$(method)_T$(T_train)_$(gen_error_m).npy")
			save_output_data(Lkl_a, "./learned_generalization_t50/kl_t_accl_$(method)_T$(T_train)_$(gen_error_m).npy")
			save_output_data(Lkl_d, "./learned_generalization_t50/kl_t_diff_$(method)_T$(T_train)_$(gen_error_m).npy")
		end



		#------IDEA TODO

		# function obtain_iso_vorticity(T, Vort, method)
		#       path = "./learned_figures/plot_vort_isosurface_$(method).png"
		#       Vort_field = 0.5 * (Vort[T,:,1].^2 .+ Vort[T,:,2].^2 .+ Vort[T,:,3].^2)
		#       r = 0.2; v = 1.3*r
		#       iso_surface_vorticity(Vort_field, r, v, path)
		# end
		#
		# include("gen_vorticity_outs.jl")
		# obtain_iso_vorticity(T, vort, method)
	# end
end
