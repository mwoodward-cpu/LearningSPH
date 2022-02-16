"""
Compared the learned Ws through simulating long time predictions and generating data.

"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff, LinearAlgebra


# method = "phys_inf"
# method = "phys_inf_theta"
# method = "phys_inf_Wliu"
# method = "phys_inf_Wab"
# method = "phys_inf_Wab_bg_pres"
# method = "phys_inf_Wab_po"
# method = "phys_inf_W2ab"
# method = "phys_inf_W2ab_bgp"
method = "phys_inf_W2ab_po"

function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end

make_dir("learned_figures"); make_dir("learned_sims")
make_dir("learned_data")


T = 250
t_save = 1   #initial time for saving
t_start = 1;

extern_f = "determistic"
IC = "dns"

θ = 0.0002;
h = 0.335;
dt = 0.04;

include("./data_loader.jl")
pos_path = "./wc_dns_4096_unif_longer_time/pos_traj_4k.npy"
vel_path = "./wc_dns_4096_unif_longer_time/vel_traj_4k.npy"
rho_path = "./wc_dns_4096_unif_longer_time/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)

traj_gt = traj_gt[t_start:end, :, :]
vels_gt = vels_gt[t_start:end, :, :]
rhos_gt = rhos_gt[t_start:end, :]


D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


println("*****************    *************")
println("running method = ", method)

function load_learned_data(method)
	data_dir_phy = "./learned_models/output_data_unif_tracers_forward_$(method)_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height4_mphys_inf_lkl_lf_klswitch0/"
	params_path = "$(data_dir_phy)/params_fin.npy"
	p_fin = npzread(params_path)
	return p_fin
end

p_fin = load_learned_data(method)


function include_sensitivity_file(method)
	include("./models/sensitivities_3d_$(method).jl")
end

if method!="truth"
	include("./sph_3d_integrator.jl")
	include_sensitivity_file(method)
	vort, accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, T)
end




#-----------Outputs

function simulate(pos, sim_time=5)
	sim_path = "./learned_sims/traj_N$(N)_T$(T)_h$(h)_$(IC)_θ$(θ)_$(method).mp4"
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


vor_path = "./learned_data/vort_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
acc_path = "./learned_data/accl_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
pos_path = "./learned_data/traj_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
vel_path = "./learned_data/vels_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"
rho_path = "./learned_data/rhos_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ)_$(method).npy"

function gen_data_files(vort, accl, traj, vels, rhos)
    println(" ****************** Saving data files ***********************")
    # npzwrite(pos_data, pos[t_save:end,:,:])
    # npzwrite(vel_data, vel[t_save:end,:,:])
	npzwrite(vor_path, vort[t_save:end,:,:])
	npzwrite(acc_path, accl[t_save:end,:,:])
	npzwrite(pos_path, traj[t_save:end,:,:])
	npzwrite(vel_path, vels[t_save:end,:,:])
	npzwrite(rho_path, rhos[t_save:end,:])
end

gen_data_files(vort, accl, traj, vels, rhos)


# end
