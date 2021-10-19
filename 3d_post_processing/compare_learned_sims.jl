"""
Compared the learned sims
"""

using Statistics, LaTeXStrings
using NPZ, Plots, Flux, QuadGK
using ForwardDiff
using BSON: @load

#uncomment to run each method, and obtain a comparison of simulations:

# method = "node"
# method = "nnsum"
# method = "rot_inv"
# method = "eos_nn"
# method = "grad_p"
# method = "Wnn"
# method = "phys_inf"
method = "truth"


T = 500
t_save = 1   #initial time for saving
t_start = 20;

IC = "Vrandn"
vmag = 1.0  #initial magnitude of velocity
# IC = "TG"

#params:
α = 1.0
β = 2.0*α  #usual value of params for alpha and β but these depend on problem
θ = 5e-1;

c = 10.0;
g = 7.0;
h = 0.335;
cdt = 0.4;
dt = cdt * h / c;


traj_gt = npzread("./data/traj_N4096_T50_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ0.5_AV_neg_rel.npy")
vels_gt = npzread("./data/vels_N4096_T50_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ0.5_AV_neg_rel.npy")
rhos_gt = npzread("./data/rhos_N4096_T50_ts1_h0.335_Vrandn_cdt0.4_c10.0_α1.0_β2.0_θ0.5_AV_neg_rel.npy")

traj_gt = traj_gt[t_start:end, :, :]
vels_gt = vels_gt[t_start:end, :, :]
rhos_gt = rhos_gt[t_start:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end

make_dir("learned_figures"); make_dir("learned_sims")
make_dir("learned_data")


if method =="node"
	height = 5;
	# nn_data_dir = "./learned_models/output_data_node_kl_Vrandn_itr2000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint150_ts20_coarse1/"
		#doesnt learn advection properly with just kl loss!
	nn_data_dir = "./learned_models/output_data_forward_node_kl_lf_Vrandn_itr801_lr0.005_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch0/"
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
	p = params(NN); n_list = floor(Int, size(p[1])[2]/(2*D))
end

if method =="nnsum"
	height = 5;
	nn_data_dir = "./learned_models/output_data_forward_nnsum_kl_lf_Vrandn_itr600_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch0/"
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="rot_inv"
	height = 5;
	nn_data_dir = "./learned_models/output_data_forward_rot_inv_kl_lf_Vrandn_itr2000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch1/"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="eos_nn"
	height = 9;
	nn_data_dir = "./learned_models/output_data_forward_eos_nn_lf_Vrandn_itr2000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height9_klswitch1/"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="grad_p"
	height = 5;
	nn_data_dir = "./learned_models/output_data_forward_grad_p_kl_lf_Vrandn_itr500_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch0/"
	c_gt = c;
	params_path = "$(nn_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
	n_params = size(p_fin)[1]
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
end

if method =="phys_inf"
	phys_data_dir = "./learned_models/output_data_forward_phys_inf_lf_Vrandn_itr3000_lr0.05_T10_D3_N4096_c10.0_α1.0_β2.0_h0.335_nball_nint200_ts20_coarse1_height5_klswitch1/"
	params_path = "$(phys_data_dir)/params_fin.npy"
	p_fin = npzread(params_path)
end

if method =="truth"
	p_fin = [c, α, β, g];
end

function include_sensitivity_file(method)
	include("./sensitivities_3d_$(method).jl")
end

if method!="truth"
	include_sensitivity_file(method)
end
if method=="truth"
	include_sensitivity_file("phys_inf")
end
include("./sph_3d_integrator.jl")


traj, vels, rhos = vel_verlet(traj_gt, vels_gt, p_fin, θ, T)




#-----------Outputs

sim_path = "./learned_sims/traj_N$(N)_T$(T)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).mp4"

function simulate(pos, sim_time=5)
    gr(size=(1000,800))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1:(T+1)
         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
         title = "Learned WCSPH_$(method): N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false)
         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red")
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end


#UAHPC seems to need this formatting
ENV["GKSwstype"]="100"
simulate(traj, 10)

pos_path = "./learned_data/traj_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).npy"
vel_path = "./learned_data/vels_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).npy"
rho_path = "./learned_data/rhos_N$(N)_T$(T)_ts$(t_save)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method).npy"

function gen_data_files(traj, vels, rhos)
    println(" ****************** Saving data files ***********************")
    # npzwrite(pos_data, pos[t_save:end,:,:])
    # npzwrite(vel_data, vel[t_save:end,:,:])
	npzwrite(pos_path, traj[t_save:end,:,:])
	npzwrite(vel_path, vels[t_save:end,:,:])
	npzwrite(rho_path, rhos[t_save:end,:])
end

gen_data_files(traj, vels, rhos)
