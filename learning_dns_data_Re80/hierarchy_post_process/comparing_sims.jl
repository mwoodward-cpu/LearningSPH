using Plots, NPZ, LaTeXStrings

mmt = "008"
include("./data_loader.jl")
t_start = 1; t_coarse = 1; l_method = "lf";

pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

gt_traj = traj_gt[t_start:t_coarse:end, :, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;
h = 0.335

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];

function load_traj_data_t20(method, T_pred, T_train)
	data_dir_phy = "./learned_data"
	tra_path = "$(data_dir_phy)/traj_Tp$(T_pred)_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method).npy"
	traj = npzread(tra_path);
	return traj
end

IC = "dns_equil"; T_pred = 500; T_train = 20;
# method="SPH-informed: Parameterized W"
# 	method_plt = "phys_inf_wab"
	traj_pred_wab = load_traj_data_t20(m_phys[1], T_pred, T_train);

IC = "dns"
# method_plt="node"
# 	method="Neural ODE"
 	traj_pred_node= load_traj_data_t20(m_nns[1], T_pred, T_train);

# method_plt="nnsum2"
# 	method="NN summand: pairwise interaction"
	traj_pred_nns = load_traj_data_t20(m_nns[2], T_pred, T_train);
# 	println(size(pr_traj))

# method_plt="rot_inv"
# 	method="SPH-informed: Rotationally invariant NN"
	traj_pred_rot = load_traj_data_t20(m_nns[3], T_pred, T_train);

# method_plt="grad_p"
# 	method="SPH-informed: Pressure gradient NN"
	traj_pred_gp = load_traj_data_t20(m_nns[5], T_pred, T_train);

# method_plt="eos_nn"
# 	method="SPH-informed: NN parameterized EoS"
	traj_pred_eos = load_traj_data_t20(m_nns[4], T_pred, T_train);

# method_plt="phys_inf"
# 	method="SPH-informed: basic"
# 	pr_traj = npzread("./learned_data/traj_Tp250_Tt30_h0.335_dns_phys_inf_theta_po_liv_Pi_lf_itr2200_lr0.02.npy");


T = size(gt_traj)[1];
N = size(gt_traj)[2];

function simulate_all(traj_pred_wab, traj_pred_node, traj_pred_nns, traj_pred_rot, traj_pred_gp, traj_pred_eos, traj_gt, sim_time=20)
    sim_path = "./learned_sims/gt_pr_all_traj_N$(N)_T$(T)_tcourse$(t_coarse).mp4"
    gr(size=(2100,1200))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
	ms_ = 6.5
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1 : T_pred
		println("time step = ", i)
         plt_w2ab = Plots.scatter(traj_pred_wab[i, 1:n_2, 1], traj_pred_wab[i, 1:n_2, 2], traj_pred_wab[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_pred_wab[i, (n_2+1):end, 1], traj_pred_wab[i, (n_2+1):end, 2], traj_pred_wab[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 title!(L"\textrm{SPH-informed:} W_2(a,b)")

		 plt_node = Plots.scatter(traj_pred_node[i, 1:n_2, 1], traj_pred_node[i, 1:n_2, 2], traj_pred_node[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_pred_node[i, (n_2+1):end, 1], traj_pred_node[i, (n_2+1):end, 2], traj_pred_node[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 title!(L"\textrm{Neural ODE}")

		 plt_nns = Plots.scatter(traj_pred_nns[i, 1:n_2, 1], traj_pred_nns[i, 1:n_2, 2], traj_pred_nns[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_pred_nns[i, (n_2+1):end, 1], traj_pred_nns[i, (n_2+1):end, 2], traj_pred_nns[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 title!(L"\textrm{NN summand: pairwise interaction}")

		 plt_rot = Plots.scatter(traj_pred_rot[i, 1:n_2, 1], traj_pred_rot[i, 1:n_2, 2], traj_pred_rot[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_pred_rot[i, (n_2+1):end, 1], traj_pred_rot[i, (n_2+1):end, 2], traj_pred_rot[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 title!(L"\textrm{SPH-informed: Rotationally invariant NN}")

		 # plt_gp = Plots.scatter(traj_pred_gp[i, 1:n_2, 1], traj_pred_gp[i, 1:n_2, 2], traj_pred_gp[i, 1:n_2, 3],
         #  		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         # Plots.scatter!(traj_pred_gp[i, (n_2+1):end, 1], traj_pred_gp[i, (n_2+1):end, 2], traj_pred_gp[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 # title!(L"\textrm{SPH-informed: Pressure gradient NN}")

		 plt_eos = Plots.scatter(traj_pred_eos[i, 1:n_2, 1], traj_pred_eos[i, 1:n_2, 2], traj_pred_eos[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_pred_eos[i, (n_2+1):end, 1], traj_pred_eos[i, (n_2+1):end, 2], traj_pred_eos[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 title!(L"\textrm{SPH-informed: NN parameterized EoS}")

		 plt_gt = Plots.scatter(traj_gt[i, 1:n_2, 1], traj_gt[i, 1:n_2, 2], traj_gt[i, 1:n_2, 3],
			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
		 Plots.scatter!(traj_gt[i, (n_2+1):end, 1], traj_gt[i, (n_2+1):end, 2], traj_gt[i, (n_2+1):end, 3], ms=ms_, color = "red")
		title!(L"\textrm{Ground Truth: DNS}")
		plt_out = plot(plt_gt, plt_w2ab, plt_eos, plt_rot, plt_nns, plt_node, layout=(2,3))
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end

function simulate(traj_pred, traj_gt, sim_time=18)
    sim_path = "./gt_pr_$(method_plt)_traj_N$(N)_T$(T)_tcourse$(t_course).mp4"
    gr(size=(1400,600))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
	ms_ = 4.5
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1 : T
		println("time step = ", i)
         plt = Plots.scatter(traj_pred[i, 1:n_2, 1], traj_pred[i, 1:n_2, 2], traj_pred[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_pred[i, (n_2+1):end, 1], traj_pred[i, (n_2+1):end, 2], traj_pred[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 title!(L"\textrm{%$method}")
		 plt2 = Plots.scatter(traj_gt[i, 1:n_2, 1], traj_gt[i, 1:n_2, 2], traj_gt[i, 1:n_2, 3],
			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
		 Plots.scatter!(traj_gt[i, (n_2+1):end, 1], traj_gt[i, (n_2+1):end, 2], traj_gt[i, (n_2+1):end, 3], ms=ms_, color = "red")
		title!(L"\textrm{Ground Truth: DNS}")
		plt_out = plot(plt, plt2, layout=(1,2))
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end

function simulate_set(traj_gt, sim_time=10)
    sim_path = "./gt_traj.mp4"
    gr(size=(700,600))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
	ms_ = 4.5
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1 : T
		println("time step = ", i)
		 Plots.scatter(traj_gt[i, 1:n_2, 1], traj_gt[i, 1:n_2, 2], traj_gt[i, 1:n_2, 3],
			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
		 Plots.scatter!(traj_gt[i, (n_2+1):end, 1], traj_gt[i, (n_2+1):end, 2], traj_gt[i, (n_2+1):end, 3], ms=ms_, color = "red")
		title!(L"\textrm{Ground Truth: DNS}")
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end

# simulate_set(gt_traj)
# simulate(pr_traj, gt_traj)
simulate_all(traj_pred_wab, traj_pred_node, traj_pred_nns, traj_pred_rot, traj_pred_gp, traj_pred_eos, gt_traj)
