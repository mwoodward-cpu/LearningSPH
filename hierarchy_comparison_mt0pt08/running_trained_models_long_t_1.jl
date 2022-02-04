"""
Using Trained models to make long time predcitions (9 times longer than training)

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra


m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi", 
          "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_Wab_theta_po"];
l_m = ["lf", "kl_lf_t", "l2"];

method = m_phys[1];
#l_method = l_m[1];
#itr = 2200; lr = 0.02; T = 30; θ0 = 0.0009;

#method = ARGS[1];
l_method = ARGS[1];
itr = parse(Int, ARGS[2]);
lr = parse(Float64, ARGS[3]);
T = parse(Int, ARGS[4]);
θ0 = 0.0009; 

include("./utils.jl")
make_dir("learned_figures"); make_dir("learned_sims")
make_dir("learned_data")

T_pred = 250
t_start = 1;

extern_f = "determistic"
IC = "dns"

h = 0.335;
t_coarse = 2;
dt = 0.02 * t_coarse;

include("./data_loader.jl")
pos_path = "./wc_dns_unif_4096_gen_data_mt0.08/pos_traj_4k.npy"
vel_path = "./wc_dns_unif_4096_gen_data_mt0.08/vel_traj_4k.npy"
rho_path = "./wc_dns_unif_4096_gen_data_mt0.08/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :];
vels_gt = vels_gt[t_start:t_coarse:end, :, :];
rhos_gt = rhos_gt[t_start:t_coarse:end, :];


D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


println("*****************    *************")
println("running method = ", method)
include("load_models.jl")
p_fin = load_phys_inf_learned_model(method, l_method, itr, lr, T, θ0, t_coarse, dt)


function include_sensitivity_file(method)
	include("./models/sensitivities_3d_$(method).jl")
end

if method!="truth"
	include_sensitivity_file(method)
	include("./sph_3d_integrator.jl")
	accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, T_pred)
end




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


acc_path = "./learned_data/accl_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
pos_path = "./learned_data/traj_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
vel_path = "./learned_data/vels_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"
rho_path = "./learned_data/rhos_Tp$(T_pred)_Tt$(T)_h$(h)_$(IC)_$(method)_$(l_method)_itr$(itr)_lr$(lr).npy"

function gen_data_files(accl, traj, vels, rhos)
    println(" ****************** Saving data files ***********************")
	# npzwrite(vor_path, vort[t_save:end,:,:])
	npzwrite(acc_path, accl)
	npzwrite(pos_path, traj)
	npzwrite(vel_path, vels)
	npzwrite(rho_path, rhos)
end

gen_data_files(accl, traj, vels, rhos)

#UAHPC seems to need this formatting
ENV["GKSwstype"]="100"
simulate(traj, 15)
if method=="truth"
	simulate(traj_gt, 15)
end
