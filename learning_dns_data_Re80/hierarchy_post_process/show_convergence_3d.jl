using Plots, NPZ, Statistics

#showing convergence over iterations.
mmt = "008";
t_start = 1; t_coarse=1;
h=0.335; extern_f = "determistic";
dt = 0.04;

include("./data_loader.jl")
pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]
D = size(traj_gt)[3];
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


include("./load_models_t20.jl")
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()

as = npzread("$(d1)/a_out.npy")
bs = npzread("$(d1)/b_out.npy")
αs = npzread("$(d1)/alpha_out.npy")
βs = npzread("$(d1)/beta_out.npy")
gs = npzread("$(d1)/g_out.npy")
cs = npzread("$(d1)/c_out.npy")
ps = npzread("$(d1)/po_out.npy")
θs = npzread("$(d1)/theta_out.npy")

# c_in, α_in, β_in, g_in, a_in, b_in, po_in, θ_in = p_in
params = cat(cs,αs,βs,gs,as,bs,ps,θs, dims=2)
method = "phys_inf_W2ab_theta_po_liv_Pi";

include("./models/sensitivities_3d_$(method).jl")
include("./sph_3d_integrator.jl")

function save_final_snaphshot(params)
    n_itrs, n_params = size(params)
    for i in 1 : 40
        println("iteration = ", i)
        accl, traj, vels, rhos = vel_verlet(traj_gt, vels_gt, params[i,:], 70)
        vf_pr = vels[end,:,:];
        Vf = reshape(vf_pr, (16, 16, 16, 3));
        Vf_u = Vf[:,:,:,1];
        npzwrite("./3d_conv_data/vf_u_t70_$(method)_itr$(i).npy", Vf_u)
    end
    println("3d_conv_data collected")
end

save_final_snaphshot(params)
