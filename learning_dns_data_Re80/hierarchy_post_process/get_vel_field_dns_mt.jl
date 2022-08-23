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


# mt = 0.16
mt = 0.04

extern_f = "determistic"
IC = "dns_equil"


θ = 0.0002;
h = 0.335;
t_coarse = 1
dt = t_coarse*0.04;
T_pred = 200; T_train = 20;


include("./data_loader.jl")


		include("./kde_G.jl")
		# include("./loss_functions_gen.jl")
		make_dir("learned_generalization");



		function load_gen_mt_gt_data(mmt)
			pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
			vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
			rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
			traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)
			# Vf_gt = npzread("./learned_field_data/vf_gt_Mt$(mt).npy")
			return traj_gt, vels_gt, rhos_gt
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

		mmt = string_mt(mt)
		traj_gt, vels_gt, rhos_gt = load_gen_mt_gt_data(mmt);
		Vf_truth = obtain_vel_mt_field_values(T_pred, traj_gt, vels_gt, rhos_gt);

		save_output_data(Vf_truth, "./learned_generalization/vf_gt_$(mmt).npy")

		println(size(Vf_truth))
