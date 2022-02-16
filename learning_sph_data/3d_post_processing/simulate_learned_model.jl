

using Flux, Statistics, LaTeXStrings
using ForwardDiff, Plots
using Zygote
using NPZ

using BSON: @load

loss_method = "kl"
# loss_method = "l2"
# loss_method = "kl_t_one_dis"
# loss_method = "kl_t"
# loss_method = "kl_l2_t"

# method = "nn_sum"
# method = "nn_sum2"
# method = "rot_inv"
method = "node"
# method = "eos_nn"
# method = "grad_p"


α = 1.0; β = 2*α; h = 0.2; g = 7; θ = 0.8e-1; c = 12.0; c_gt = c;
cdt = 0.4;
dt = cdt * h/c

# ic_data_dir = "/home/adele/sph_learning/analytic_gradient_method/physics_informed_sph_learning"

t_start = 3205;

traj_gt = npzread("./data/traj_N1024_T6001_ts5801_h0.2_s_hit_cdt0.4_c12.0_α1.0_β2.0_θ0.08_AV_neg_rel_ke.npy")
vels_gt = npzread("./data/vels_N1024_T6001_ts5801_h0.2_s_hit_cdt0.4_c12.0_α1.0_β2.0_θ0.08_AV_neg_rel_ke.npy")
rhos_gt = npzread("./data/rhos_N1024_T6001_ts5801_h0.2_s_hit_cdt0.4_c12.0_α1.0_β2.0_θ0.08_AV_neg_rel_ke.npy")

traj_gt = traj_gt[t_start:end, :, :]
vels_gt = vels_gt[t_start:end, :, :]
rhos_gt = rhos_gt[t_start:end, :]


N = size(traj_gt)[2];
D = size(traj_gt)[3];
m = (2. * pi)^D / N; #so that ρ₀ = 1;


if method == "node"
	height = 3;
	nn_data_dir = "./semi_inf_data4/output_data_forward_kl_node_s_hit_itr2000_lr0.05_T3_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt3_coarse5"
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_)[1]
	p = params(NN); n_list = floor(Int, size(p[1])[2]/4)
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	include("sensitivities_node.jl")
end


if method == "rot_inv"
	height = 3;
	if loss_method=="kl_t"
		nn_data_dir = "./semi_inf_data/output_data_kl_t_rot_inv_s_hit_itr2000_lr0.05_T3_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt3_coarse5"
	end
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_)[1]
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	include("sensitivities_rot_nn.jl")
end

if method =="nn_sum"
	height = 3;
	if loss_method=="kl_t"
		nn_data_dir = "./semi_inf_data/output_data_kl_t_nn_sum_s_hit_itr2000_lr0.05_T3_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt3_coarse5"
	end
	if loss_method=="kl_l2_t"
		nn_data_dir = "./semi_inf_data/output_data_kl_l2_t_nn_sum_s_hit_itr2000_lr0.05_T3_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt3_coarse5"
	end
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_)[1]
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	include("sensitivities_nnsum.jl")
end


if method =="nn_sum2"
	nn_data_dir = "./semi_inf_data/output_data_kl_nn_sum2_s_hit_itr2000_lr0.04_T5_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt3_coarse3"
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_)[1]
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	include("sensitivities_nnsum2.jl")
end


if method =="grad_p"
	nn_data_dir = "./optimiz_semi_inf/output_data_kl_grad_p_s_hit_itr1800_lr0.05_T2_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt3"
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_)[1]
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	include("sensitivities_grad_p.jl")
end

if method =="eos_nn"
	height = 8;
	# nn_data_dir = "./semi_inf_data/output_data_l2_eos_nn_s_hit_itr2000_lr0.05_T3_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt8_coarse5"
	nn_data_dir = "./semi_inf_data/output_data_kl_t_one_dist_eos_nn_s_hit_itr2000_lr0.05_T5_D2_N1024_c12.0_α1.0_β2.0_h0.2_nint90_ts3205_hgt8_coarse3"
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_)[1]
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	include("sensitivities_eos_nn.jl")
end


include("./sph_simulator.jl")
T = 500;
traj, vels, rhos = vel_verlet_NN(traj_gt, vels_gt, p_fin, α, β, h, c, g, θ, T)


function obtain_tke(vels)
  tke = zeros(T);
  for k in 1 : T
      tke[k] = 0.5*mean(vels[k,:,1].^2 .+ vels[k,:,2].^2)
  end
  return tke
end

tke = obtain_tke(vels)

function compute_Re(vels)
  Re = zeros(T+1)
  V = 0.0
  L = 2*pi
  ν = 1/10 * α * c * h
  for t in 1 : (T+1)
    V = maximum(vels[t, :, :])
    Re[t] = L*V/ν
  end
  return Re
end

Re = compute_Re(vels)

function compute_turb_ke(vels)
	turb_ke = zeros(T);
	for t in 1 : T
		avg_KE = 0.5*mean(vels[t, :, 1].^2 .+ vels[t, :, 2].^2)
		dec_ke = 0.5 * (mean(vels[t, :, 1])^2 + mean(vels[t, :, 2])^2)
		turb_ke[t] = avg_KE - dec_ke
	end
	return turb_ke
end

turb_ke = compute_turb_ke(vels)

#-----outputs
function create_dir(dir_name)
	if isdir(dir_name) == true
	    println("directory already exists")
	else mkdir(dir_name)
	end
end

create_dir("mod_learned_figures"); create_dir("sims_learned");

ENV["GKSwstype"]="100"


#-----------Outputs

sim_path = "./sims_learned/traj_N$(N)_T$(T)_h$(h)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method)_$(loss_method).mp4"
file_out_tke = "./mod_learned_figures/tke_N$(N)_T$(T)_h$(h)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method)_$(loss_method).png"
file_out_re = "./mod_learned_figures/re_N$(N)_T$(T)_h$(h)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method)_$(loss_method).png"
file_out_turb_ke = "./mod_learned_figures/turb_ke_N$(N)_T$(T)_h$(h)_cdt$(cdt)_c$(c)_α$(α)_β$(β)_θ$(θ)_$(method)_$(loss_method).png"


function plot_KE_fluc()
    gr(size=(500,400))
    println("*************** generating plots ******************")

	plt = plot(turb_ke, label=L"k", color="blue", linewidth = 2.25)
	title!(L"\textrm{Turbulent kinetic energy: } \theta = %$(θ)", titlefont=16)
	xlabel!(L"t", xtickfontsize=10, xguidefontsize=16)
	ylabel!(L"k", ytickfontsize=10, yguidefontsize=16)
    display(plt)
    savefig(plt, file_out_turb_ke)
end


function plotting_Re()
  gr(size=(500,500))
  plt = plot(Re, label=L"Re", color="blue", linewidth = 2.25)
  title!(L"\textrm{Reynolds number: } \theta = %$(θ)", titlefont=16)
  xlabel!(L"t", xtickfontsize=10, xguidefontsize=16)
  ylabel!(L"Re", ytickfontsize=10, yguidefontsize=16)
  display(plt)
  savefig(plt, file_out_re)
end




function simulate(pos, sim_time=10)
    gr(size=(1000,800))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1:2:(T+1)
         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2],
         title = "WCSPH_$(method): N=$(N), h=$(h), c=$(c)", xlims = [0, 2*pi], ylims = [0,2*pi], legend = false)
         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], color = "red")
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end


function simulate_same_color(pos, sim_time=8)
    gr(size=(1000,800))
    println("**************** Simulating the particle flow ***************")
    #theme(:juno)
    n_2 = round(Int,N/2)
    anim = @animate for i ∈ 1:2:(T+1)
         Plots.scatter(pos[i, :, 1], pos[i, :, 2],
         title = "WCSPH_$(method): N=$(N), h=$(h), c=$(c)", xlims = [0, 2*pi],
         ylims = [0,2*pi], legend = false, color="blue")
    end
    gif(anim, sim_path, fps = round(Int, T/sim_time))
    println("****************  Simulation COMPLETE  *************")
end


plot_KE_fluc()
plotting_Re()
simulate(traj, 10)
# simulate_same_color(traj, 8)
