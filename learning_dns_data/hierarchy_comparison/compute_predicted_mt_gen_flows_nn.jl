"""
Using Trained models to make long time predcitions (9 times longer than training)

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra

Mt = 0.16;


m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi",
          "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_Wab_theta_po"];
l_m = ["lf", "kl_lf_t", "l2"];
m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];

# method = m_phys[1];
method = m_nns[5];

l_method = l_m[1];
T = 30;

function obtain_itr_lr(method)
	if method =="node_norm_theta_liv_Pi"
		itr = 500; lr = 0.002; θ0 = 0.0012; class = "node"; height = 6
	end
	if method =="nnsum2_norm_theta_liv_Pi"
		itr = 600; lr = 0.01; θ0 = 0.0012; class = "nnsum"; height = 6
	end
	if method =="rot_inv_theta_liv_Pi"
		itr = 500; lr = 0.02; θ0 = 0.0009; class = "rot"; height = 6
	end
	if method =="grad_p_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02; θ0 = 0.0009; class = "gradp"; height = 6
	end
	if method =="eos_nn_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02; θ0 = 0.0009; class = "eos"; height = 8
	end
	return itr, lr, θ0, class, height
end

# itr, lr, θ0, class, height = obtain_itr_lr(method);
#method = ARGS[1];
# l_method = ARGS[1];
# itr = parse(Int, ARGS[2]);
# lr = parse(Float64, ARGS[3]);
# T = parse(Int, ARGS[4]);
c_gt = 0.845; g = 1.4; c = c_gt;


include("./utils.jl")
make_dir("learned_figures"); make_dir("learned_sims")
make_dir("learned_data")

T_pred = 30;
t_start = 1;

extern_f = "determistic"
IC = "dns"

h = 0.335;
t_coarse = 2;
dt = 0.02 * t_coarse;

include("./data_loader.jl")
pos_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/pos_traj_4k.npy"
vel_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/vel_traj_4k.npy"
rho_path = "./wc_dns_unif_4096_gen_data_mt$(Mt)/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :];
vels_gt = vels_gt[t_start:t_coarse:end, :, :];
rhos_gt = rhos_gt[t_start:t_coarse:end, :];


D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


include("load_models.jl")
# p_fin = load_phys_inf_learned_model(method, l_method, itr, lr, T, θ0, t_coarse, dt)
# p_fin, NN, re = load_nn_learned_model(method, l_method, height, class, itr, lr, T, θ0, t_coarse, dt)

h_kde = 0.9


function include_sensitivity_file(method)
	include("./models/sensitivities_3d_$(method).jl")
end

	println("*****************    *************")
	println("running method = ", method)
	itr, lr, θ0, class, height = obtain_itr_lr(method);
	p_fin, NN, re = load_nn_learned_model(method, l_method, height, class, itr, lr, T, θ0, t_coarse, dt)
	println(p_fin[end])
	if Mt == 0.04
		p_fin[end] = 0.01 * p_fin[end];
	end

	if Mt == 0.16
		p_fin[end] = 2*p_fin[end];
	end
	println(p_fin[end])

	if method!="truth"
		include_sensitivity_file(method)
		include("./sph_3d_integrator.jl")
		accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, T_pred)
	end


	function compute_gt_accl(vel_gt)
		accl_gt = zeros(T, N, D);
		for t in 1 : (T-1)
			for i in 1 : D
				accl_gt[t+1,:,i] = (vel_gt[t+1,:,i] .- vel_gt[t,:,i]) / dt
			end
		end
		return accl_gt
	end

	include("kde_G.jl")

	accl_gt = compute_gt_accl(vels_gt)

	function comparing_Ga(width=0.22)
	    gr(size=(700,600))
	    x_s = -width
	    x_e = width
		Ga_m(x) = kde(x, vec(accl[T, :, :]));
		Ga_gt(x) = kde(x, vec(accl_gt[T, :, :]));
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
	    # out_path = "./learned_figures/Ga_trained_comp_t$(T).png"
		#
	    # savefig(plt, out_path)
	end

	if Mt==0.04
		win = 0.03
	end

	if Mt==0.08
		win = 0.1
	end

	if Mt==0.16
		win = 0.2
	end
	comparing_Ga(win)
	function gen_data_files(accl, traj, vels, rhos)
		acc_path = "./learned_data/accl_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
		pos_path = "./learned_data/traj_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
		vel_path = "./learned_data/vels_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"
		rho_path = "./learned_data/rhos_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr)_Mt$(Mt).npy"

	    println(" ****************** Saving data files ***********************")
		# npzwrite(vor_path, vort[t_save:end,:,:])
		npzwrite(acc_path, accl)
		npzwrite(pos_path, traj)
		npzwrite(vel_path, vels)
		npzwrite(rho_path, rhos)
	end

	gen_data_files(accl, traj, vels, rhos)


#-----------Outputs

function simulate(pos, sim_time=15)
    sim_path = "./learned_sims/traj_N$(N)_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr).mp4"
    gr(size=(1000,800))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1 : T_pred
		println("time step = ", i)
         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
         title = "Learned WCSPH_$(method): N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false)
         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red")
    end
    gif(anim, sim_path, fps = round(Int, T_pred/sim_time))
    println("****************  Simulation COMPLETE  *************")
end


#UAHPC seems to need this formatting
# ENV["GKSwstype"]="100"
# simulate(traj, 10)
# if method=="truth"
# 	simulate(traj_gt, 10)
# end
