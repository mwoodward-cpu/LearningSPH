"""
Compared the learned models through simulating long time predictions and generating data.

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra

function make_dir(path)
		if isdir(path) == true
			   println("directory already exists")
		   else mkdir(path)
		end
end
make_dir("learned_figures"); make_dir("learned_sims")
make_dir("learned_data")

T = 250
t_start = 1;

extern_f = "determistic"
IC = "dns"

h = 0.335;
t_coarse = 2;
dt = 0.04;




m_phys = ["phys_inf_theta", "phys_inf_theta_livescu_ext", "phys_inf_theta_correct_Pi", "phys_inf_Wab_theta",
			"phys_inf_W2ab_theta", "phys_inf_Wliu_theta", "phys_inf_theta_po", "phys_inf_Wab_po_theta"];

m_nn = ["node_norm", "nnsum2_norm_theta", "grad_p_theta", "eos_nn_theta", "rot_inv"];

l_m = ["lf", "kl_lf"];

method = m_nn[2];
l_method = l_m[1];
# l_method = ARGS[1];


include("./data_loader.jl")
N = 4096; D = 3; m = (2.0 * pi)^D / N;

# for mt in [0.04, 0.08, 0.16]
mt = 0.16

	pos_path = "./wc_dns_unif_4096_gen_data_mt$(mt)/pos_traj_4k.npy"
	vel_path = "./wc_dns_unif_4096_gen_data_mt$(mt)/vel_traj_4k.npy"
	rho_path = "./wc_dns_unif_4096_gen_data_mt$(mt)/rho_traj_4k.npy"
	traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)

	traj_gt = traj_gt[t_start:t_coarse:end, :, :]
	vels_gt = vels_gt[t_start:t_coarse:end, :, :]
	rhos_gt = rhos_gt[t_start:t_coarse:end, :]


	println("*****************    *************")
	println("running method = ", method)
	include("load_models.jl")
	height, itr, lr, kl_s = obtain_height_itr(method)
	p_fin, NN, re = load_nn_learned_model(method, height, itr, lr, kl_s)


	function include_sensitivity_file(method)
		include("./models/sensitivities_3d_$(method).jl")
	end

	if method!="truth"
		include_sensitivity_file(method)
		include("./sph_3d_integrator.jl")
		accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, T)
	end




	#-----------Outputs

	function simulate(pos, sim_time=5)
	    sim_path = "./learned_sims/traj_N$(N)_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(mt).mp4"
	    gr(size=(1000,800))
	    println("**************** Simulating the particle flow ***************")
	    #theme(:juno)
	    n_2 = round(Int,N/2)
	    anim = @animate for i ∈ 1:T
			println("time step = ", i)
	         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
	         title = "Learned WCSPH_$(method): N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false)
	         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red")
	    end
	    gif(anim, sim_path, fps = round(Int, T/sim_time))
	    println("****************  Simulation COMPLETE  *************")
	end


	#UAHPC seems to need this formatting
	ENV["GKSwstype"]="100"
	simulate(traj, 12)
	if method=="truth"
		simulate(traj_gt, 12)
	end


	# vor_path = "./learned_data/vort_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
	acc_path = "./learned_data/accl_N$(N)_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(mt).npy"
	pos_path = "./learned_data/traj_N$(N)_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(mt).npy"
	vel_path = "./learned_data/vels_N$(N)_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(mt).npy"
	rho_path = "./learned_data/rhos_N$(N)_T$(T)_h$(h)_$(IC)_$(method)_$(l_method)_mt$(mt).npy"

	function gen_data_files(accl, traj, vels, rhos)
	    println(" ****************** Saving data files ***********************")
		# npzwrite(vor_path, vort[t_save:end,:,:])
		npzwrite(acc_path, accl)
		npzwrite(pos_path, traj)
		npzwrite(vel_path, vels)
		npzwrite(rho_path, rhos)
	end

	gen_data_files(accl, traj, vels, rhos)

# end
