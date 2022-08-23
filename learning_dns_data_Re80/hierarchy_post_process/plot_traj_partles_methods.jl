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
h = 0.335;

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
 	traj_pred_node = load_traj_data_t20(m_nns[1], T_pred, T_train);

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
dt = 0.04;

function plot_traj_all(traj_in, method, latex_method, t)
    plt_path = "./learned_figures/gt_pr_traj_t_N$(N)_T$(T)_$(method)_tcourse$(t_coarse)_t$(t).png"
    gr(size=(3600,500))
    println("**************** Simulating the particle flow ***************")
	ms_ = 5.2; mult = 2.5;
    n_2 = round(Int,N/2)
    i = t;
		println("time step = ", i); t1 = round(dt*i, digits=1)
		plt0 = Plots.scatter(traj_in[1, 1:n_2, 1], traj_in[1, 1:n_2, 2], traj_in[1, 1:n_2, 3],
			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_,
			   zlabel=latex_method, ztickfontsize=1, zguidefontsize=30, legend = false)
		Plots.scatter!(traj_in[1, (n_2+1):end, 1], traj_in[1, (n_2+1):end, 2], traj_in[1, (n_2+1):end, 3], ms=ms_, color = "red")
		if method=="dns" title!(L"\textbf{t = 0 s}", titlefont=30) end

		# gr(size=(2800,500))
         plt1 = Plots.scatter(traj_in[i, 1:n_2, 1], traj_in[i, 1:n_2, 2], traj_in[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[i, (n_2+1):end, 1], traj_in[i, (n_2+1):end, 2], traj_in[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 if method=="dns" title!(L"\textbf{t = %$(t1) s}", titlefont=30) end

		 t2 = round(mult*2*dt*i, digits=1); t2_idx = round(Int, t2/dt);
		 plt2 = Plots.scatter(traj_in[t2_idx, 1:n_2, 1], traj_in[t2_idx, 1:n_2, 2], traj_in[t2_idx, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[t2_idx, (n_2+1):end, 1], traj_in[t2_idx, (n_2+1):end, 2], traj_in[t2_idx, (n_2+1):end, 3], ms=ms_, color = "red")
		 if method=="dns"  title!(L"\textbf{t = %$(t2) s}", titlefont=30) end

		 t3 = round(mult*3*dt*i, digits=1); t3_idx = round(Int, t3/dt);
		 plt3 = Plots.scatter(traj_in[t3_idx, 1:n_2, 1], traj_in[t3_idx, 1:n_2, 2], traj_in[t3_idx, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[t3_idx, (n_2+1):end, 1], traj_in[t3_idx, (n_2+1):end, 2], traj_in[t3_idx, (n_2+1):end, 3], ms=ms_, color = "red")
		 if method=="dns"  title!(L"\textbf{t = %$(t3) s}", titlefont=30) end

		 t4 = round(mult*4*dt*i, digits=1); t4_idx = round(Int, t4/dt);
		 plt4 = Plots.scatter(traj_in[t4_idx, 1:n_2, 1], traj_in[t4_idx, 1:n_2, 2], traj_in[t4_idx, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[t4_idx, (n_2+1):end, 1], traj_in[t4_idx, (n_2+1):end, 2], traj_in[t4_idx, (n_2+1):end, 3], ms=ms_, color = "red")
		 if method=="dns" title!(L"\textbf{t = %$(t4) s}", titlefont=30) end


		plt_out = plot(plt0, plt1, plt2, plt3, plt4, layout=(1,5))
	display(plt_out)
	savefig(plt_out, plt_path)
end

mc = [L"\textbf{NODE}" L"\textbf{\sum NN}" L"\textbf{Rot-Inv}" L"\textbf{(\nabla P)_{nn}}" L"\textbf{P_{nn}(\rho)}" L"\textbf{Phys-Inf: } W_{2}"] # L"W_{ab; p_0, \theta}"];

plot_traj_all(traj_gt, "dns", L"\textbf{DNS: Truth}", 50)
plot_traj_all(traj_pred_wab, m_phys[1], mc[end], 50)
plot_traj_all(traj_pred_node, m_nns[1], mc[1], 50)
plot_traj_all(traj_pred_nns, m_nns[2], mc[2], 50)
plot_traj_all(traj_pred_rot, m_nns[3], mc[3], 50)
plot_traj_all(traj_pred_eos, m_nns[5], mc[4], 50)
plot_traj_all(traj_pred_gp, m_nns[4], mc[5], 50)




function plot_traj_all(traj_in, method, latex_method, t)
    plt_path = "./learned_figures/gt_pr_traj_t_N$(N)_T$(T)_$(method)_tcourse$(t_coarse)_t$(t).png"
    gr(size=(2800,500))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
	ms_ = 5.8; mult = 2.5;
    n_2 = round(Int,N/2)
    i = t;
		println("time step = ", i); t1 = round(dt*i, digits=1)
		plt0 = Plots.scatter(traj_in[1, 1:n_2, 1], traj_in[1, 1:n_2, 2], traj_in[1, 1:n_2, 3],
			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_,
			   zlabel=L"\textbf{%$latex_method}", ztickfontsize=1, zguidefontsize=30, legend = false)
		Plots.scatter!(traj_in[1, (n_2+1):end, 1], traj_in[1, (n_2+1):end, 2], traj_in[1, (n_2+1):end, 3], ms=ms_, color = "red")
		# title!(L"\textbf{t = 0 s}", titlefont=30)

		# gr(size=(2800,500))
         plt1 = Plots.scatter(traj_in[i, 1:n_2, 1], traj_in[i, 1:n_2, 2], traj_in[i, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[i, (n_2+1):end, 1], traj_in[i, (n_2+1):end, 2], traj_in[i, (n_2+1):end, 3], ms=ms_, color = "red")
		 # title!(L"\textbf{t = %$(t1) s}", titlefont=30)

		 t2 = round(mult*2*dt*i, digits=1); t2_idx = round(Int, t2/dt);
		 plt2 = Plots.scatter(traj_in[t2_idx, 1:n_2, 1], traj_in[t2_idx, 1:n_2, 2], traj_in[t2_idx, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[t2_idx, (n_2+1):end, 1], traj_in[t2_idx, (n_2+1):end, 2], traj_in[t2_idx, (n_2+1):end, 3], ms=ms_, color = "red")
		 # title!(L"\textbf{t = %$(t2) s}", titlefont=30)

		 t3 = round(mult*3*dt*i, digits=1); t3_idx = round(Int, t3/dt);
		 plt3 = Plots.scatter(traj_in[t3_idx, 1:n_2, 1], traj_in[t3_idx, 1:n_2, 2], traj_in[t3_idx, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[t3_idx, (n_2+1):end, 1], traj_in[t3_idx, (n_2+1):end, 2], traj_in[t3_idx, (n_2+1):end, 3], ms=ms_, color = "red")
		 # title!(L"\textbf{t = %$(t3) s}", titlefont=30)

		 t4 = round(mult*4*dt*i, digits=1); t4_idx = round(Int, t4/dt);
		 plt4 = Plots.scatter(traj_in[t4_idx, 1:n_2, 1], traj_in[t4_idx, 1:n_2, 2], traj_in[t4_idx, 1:n_2, 3],
          		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
         Plots.scatter!(traj_in[t4_idx, (n_2+1):end, 1], traj_in[t4_idx, (n_2+1):end, 2], traj_in[t4_idx, (n_2+1):end, 3], ms=ms_, color = "red")
		 # title!(L"\textbf{t = %$(t4) s}", titlefont=30)

		plt_out = plot(plt0, plt1, plt2, plt3, plt4, layout=(1,5))

	display(plt_out)
	savefig(plt_out, plt_path)
end


function obtain_snapshots2(traj, n_snaps=4)
        m_s = 1.2
        ratio = 1/(n_snaps+2); horz = 1500; vert = ceil(Int, ratio*horz);
        gr(size=(horz,vert))
        n_2 = round(Int,N/2);
        t_steps = size(traj)[1]; T = t_steps - 1;
        t_range = ceil.(Int, range(12, ceil(Int, T), length=n_snaps));

        p0 = plot(zeros(2,2), xlims = [0, 1], ylims = [0,1], axis=([], false), grid = false)

        p1 =Plots.scatter(traj[1, 1:n_2, 1], traj[1, 1:n_2, 2], traj[1, 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s,
                zlabel=L"\textrm{%$latex_method}", ztickfontsize=1, zguidefontsize=22)
                # zlabel=L"d_t v_i = NN_{\theta}", ztickfontsize=1, zguidefontsize=30)
                Plots.scatter!(traj[1, (n_2+1):end, 1], traj[1, (n_2+1):end, 2],
                traj[t_range[1],(n_2+1):end, 3], color = "red", ms=m_s, title=L"t_0")


        p2 =Plots.scatter(traj[t_range[1], 1:n_2, 1], traj[t_range[1], 1:n_2, 2], traj[t_range[1], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[1], (n_2+1):end, 1], traj[t_range[1], (n_2+1):end, 2],
                traj[t_range[1],(n_2+1):end, 3], color = "red", ms=m_s, title=L"t_{\lambda}")

        p3 =Plots.scatter(traj[t_range[2], 1:n_2, 1], traj[t_range[2], 1:n_2, 2], traj[t_range[2], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[2], (n_2+1):end, 1], traj[t_range[2], (n_2+1):end, 2],
                traj[t_range[2], (n_2+1):end, 3], color = "red", ms=m_s, title=L"t_2")

        p4 =Plots.scatter(traj[t_range[3], 1:n_2, 1], traj[t_range[3], 1:n_2, 2], traj[t_range[3], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[3], (n_2+1):end, 1], traj[t_range[3], (n_2+1):end, 2],
                traj[t_range[3], (n_2+1):end, 3], color = "red", ms=m_s, title=L"t_3")

        p5 =Plots.scatter(traj[t_range[4], 1:n_2, 1], traj[t_range[4], 1:n_2, 2], traj[t_range[4], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[4], (n_2+1):end, 1], traj[t_range[4], (n_2+1):end, 2],
                 traj[t_range[4], (n_2+1):end, 3], color = "red", ms=m_s, title=L"\mathcal{O}(t_{eddy})")


        if n_snaps==5
        p6 =Plots.scatter(traj[t_range[5], 1:n_2, 1], traj[t_range[5], 1:n_2, 2], traj[t_range[5], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[5], (n_2+1):end, 1], traj[t_range[5], (n_2+1):end, 2],
                traj[t_range[5], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_{eddy}")

                plt = plot(p1, p2, p3, p4, p5, p6, layout = (1, 6), legend = false)
        end
        if n_snaps==4
                plt = plot(p0, p1, p2, p3, p4, p5, layout = (1, 6), legend = false)
        end
        # title!("Learned WCSPH_$(method): N=$(N)")
        out_path = "./learned_figures/gen_t_snapshots_N$(N)_T$(T)_h$(h)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).png"
        display(plt)
        savefig(plt, out_path)
end




# function simulate_all(traj_pred_wab, traj_pred_node, traj_pred_nns, traj_pred_rot, traj_pred_gp, traj_pred_eos, traj_gt, sim_time=20)
#     sim_path = "./learned_sims/gt_pr_all_traj_N$(N)_T$(T)_tcourse$(t_coarse).mp4"
#     gr(size=(2100,1200))
#     println("**************** Simulating the particle flow ***************")
#     #theme(:juno)
# 	ms_ = 6.5
#     n_2 = round(Int,N/2)
#     anim = @animate for i ∈ 1 : T_pred
# 		println("time step = ", i)
#          plt_w2ab = Plots.scatter(traj_pred_wab[i, 1:n_2, 1], traj_pred_wab[i, 1:n_2, 2], traj_pred_wab[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_wab[i, (n_2+1):end, 1], traj_pred_wab[i, (n_2+1):end, 2], traj_pred_wab[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textrm{SPH-informed:} W_2(a,b)")
#
# 		 plt_node = Plots.scatter(traj_pred_node[i, 1:n_2, 1], traj_pred_node[i, 1:n_2, 2], traj_pred_node[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_node[i, (n_2+1):end, 1], traj_pred_node[i, (n_2+1):end, 2], traj_pred_node[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textrm{Neural ODE}")
#
# 		 plt_nns = Plots.scatter(traj_pred_nns[i, 1:n_2, 1], traj_pred_nns[i, 1:n_2, 2], traj_pred_nns[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_nns[i, (n_2+1):end, 1], traj_pred_nns[i, (n_2+1):end, 2], traj_pred_nns[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textrm{NN summand: pairwise interaction}")
#
# 		 plt_rot = Plots.scatter(traj_pred_rot[i, 1:n_2, 1], traj_pred_rot[i, 1:n_2, 2], traj_pred_rot[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_rot[i, (n_2+1):end, 1], traj_pred_rot[i, (n_2+1):end, 2], traj_pred_rot[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textrm{SPH-informed: Rotationally invariant NN}")
#
# 		 # plt_gp = Plots.scatter(traj_pred_gp[i, 1:n_2, 1], traj_pred_gp[i, 1:n_2, 2], traj_pred_gp[i, 1:n_2, 3],
#          #  		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          # Plots.scatter!(traj_pred_gp[i, (n_2+1):end, 1], traj_pred_gp[i, (n_2+1):end, 2], traj_pred_gp[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 # title!(L"\textrm{SPH-informed: Pressure gradient NN}")
#
# 		 plt_eos = Plots.scatter(traj_pred_eos[i, 1:n_2, 1], traj_pred_eos[i, 1:n_2, 2], traj_pred_eos[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_eos[i, (n_2+1):end, 1], traj_pred_eos[i, (n_2+1):end, 2], traj_pred_eos[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textrm{SPH-informed: NN parameterized EoS}")
#
# 		 plt_gt = Plots.scatter(traj_gt[i, 1:n_2, 1], traj_gt[i, 1:n_2, 2], traj_gt[i, 1:n_2, 3],
# 			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
# 		 Plots.scatter!(traj_gt[i, (n_2+1):end, 1], traj_gt[i, (n_2+1):end, 2], traj_gt[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		title!(L"\textrm{Ground Truth: DNS}")
# 		plt_out = plot(plt_gt, plt_w2ab, plt_eos, plt_rot, plt_nns, plt_node, layout=(2,3))
#     end
#     gif(anim, sim_path, fps = round(Int, T/sim_time))
#     println("****************  Simulation COMPLETE  *************")
# end
#
#
# # simulate_set(gt_traj)
# # simulate(pr_traj, gt_traj)
# # simulate_all(traj_pred_wab, traj_pred_node, traj_pred_nns, traj_pred_rot, traj_pred_gp, traj_pred_eos, gt_traj)
#
# function plot_traj_all(traj_pred_wab, traj_pred_node, traj_pred_nns, traj_pred_rot, traj_pred_gp, traj_pred_eos, traj_gt, t)
#     plt_path = "./learned_figures/gt_pr_all_traj_N$(N)_T$(T)_tcourse$(t_coarse)_t$(t).png"
#     gr(size=(3600,600))
#     println("**************** Simulating the particle flow ***************")
#     #theme(:juno)
# 	ms_ = 6.0
#     n_2 = round(Int,N/2)
#     i = t;
# 		println("time step = ", i)
#          plt_w2ab = Plots.scatter(traj_pred_wab[i, 1:n_2, 1], traj_pred_wab[i, 1:n_2, 2], traj_pred_wab[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_wab[i, (n_2+1):end, 1], traj_pred_wab[i, (n_2+1):end, 2], traj_pred_wab[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textbf{W_2(a,b)}", titlefont=24)
#
# 		 plt_node = Plots.scatter(traj_pred_node[i, 1:n_2, 1], traj_pred_node[i, 1:n_2, 2], traj_pred_node[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_node[i, (n_2+1):end, 1], traj_pred_node[i, (n_2+1):end, 2], traj_pred_node[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textbf{NODE}", titlefont=24)
#
# 		 plt_nns = Plots.scatter(traj_pred_nns[i, 1:n_2, 1], traj_pred_nns[i, 1:n_2, 2], traj_pred_nns[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_nns[i, (n_2+1):end, 1], traj_pred_nns[i, (n_2+1):end, 2], traj_pred_nns[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textbf{NN sum}", titlefont=24)
#
# 		 plt_rot = Plots.scatter(traj_pred_rot[i, 1:n_2, 1], traj_pred_rot[i, 1:n_2, 2], traj_pred_rot[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_rot[i, (n_2+1):end, 1], traj_pred_rot[i, (n_2+1):end, 2], traj_pred_rot[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textbf{Rot-Inv}", titlefont=24)
#
# 		 # plt_gp = Plots.scatter(traj_pred_gp[i, 1:n_2, 1], traj_pred_gp[i, 1:n_2, 2], traj_pred_gp[i, 1:n_2, 3],
#          #  		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          # Plots.scatter!(traj_pred_gp[i, (n_2+1):end, 1], traj_pred_gp[i, (n_2+1):end, 2], traj_pred_gp[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 # title!(L"\textrm{SPH-informed: Pressure gradient NN}")
#
# 		 plt_eos = Plots.scatter(traj_pred_eos[i, 1:n_2, 1], traj_pred_eos[i, 1:n_2, 2], traj_pred_eos[i, 1:n_2, 3],
#           		xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, legend = false)
#          Plots.scatter!(traj_pred_eos[i, (n_2+1):end, 1], traj_pred_eos[i, (n_2+1):end, 2], traj_pred_eos[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		 title!(L"\textbf{P_{nn}(\rho)}", titlefont=24)
#
# 		 plt_gt = Plots.scatter(traj_gt[i, 1:n_2, 1], traj_gt[i, 1:n_2, 2], traj_gt[i, 1:n_2, 3],
# 			   xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], ms=ms_, zlabel=L"z", legend = false)
# 		 Plots.scatter!(traj_gt[i, (n_2+1):end, 1], traj_gt[i, (n_2+1):end, 2], traj_gt[i, (n_2+1):end, 3], ms=ms_, color = "red")
# 		# title!(L"\textbf{DNS ~ at ~ } \textbf{t = %$(dt*t) s}", titlefont=24)
# 		title!(L"\textbf{DNS} ", titlefont=24)
# 		# xlabel!(L"\textbf{t = %$(dt*t) s}", xguidefontsize=18);
# 		plt_out = plot(plt_gt, plt_w2ab, plt_eos, plt_rot, plt_nns, plt_node, layout=(1,6))
# 	display(plt_out)
# 	savefig(plt_out, plt_path)
# end
# at_t = 100
