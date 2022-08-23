using NPZ, LaTeXStrings

l_method = "lf"; IC = "dns_equil"; h=0.335; T_pred = 200; T_train = 20;
mt = 0.04; mmt = "004"
# mt = 0.16; mmt = "016"

# function load_data_t20(method, T_pred, T_train)
#     data_dir_phy = "./learned_data_mt_relation"
#     acc_path = "$(data_dir_phy)/accl_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
#     tra_path = "$(data_dir_phy)/traj_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
#     vel_path = "$(data_dir_phy)/vels_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
#     rho_path = "$(data_dir_phy)/rhos_Tt$(T_train)_Tp$(T_pred)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
#     accl = npzread(acc_path);
#     traj = npzread(tra_path);
#     vels = npzread(vel_path);
#     rhos = npzread(rho_path);
#     return accl, traj, vels, rhos
# end

function load_data_t20(method, T_pred, T_train)
    data_dir_phy = "./learned_data_mt_relation"
    acc_path = "$(data_dir_phy)/accl_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
    tra_path = "$(data_dir_phy)/traj_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
    vel_path = "$(data_dir_phy)/vels_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
    rho_path = "$(data_dir_phy)/rhos_Tt$(T_train)_h$(h)_$(IC)_$(method)_$(l_method)_Mt$(mt).npy"
    accl = npzread(acc_path);
    traj = npzread(tra_path);
    vels = npzread(vel_path);
    rhos = npzread(rho_path);
    return accl, traj, vels, rhos
end

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
            "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];
m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
          "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];

m_tot = vcat(m_phys, m_nns);

include("./data_loader.jl")
function load_gen_mt_gt_data(mmt)
    pos_path = "./equil_ic_data/mt$(mmt)/pos_traj_4k_unif.npy"
    vel_path = "./equil_ic_data/mt$(mmt)/vel_traj_4k_unif.npy"
    rho_path = "./equil_ic_data/mt$(mmt)/rho_traj_4k_unif.npy"
    traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)
    # Vf_gt = npzread("./learned_field_data/vf_gt_Mt$(mt).npy")
    return traj_gt, vels_gt, rhos_gt
end

traj_gt, vels_gt, rhos_gt = load_gen_mt_gt_data(mmt)
D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;

include("./loss_functions.jl")
function obtain_vel_mt_field_values(t, traj_in, vels_in, rhos_in)
    Vf,d1_,d2_,d3_,d4_,d5_,d6_ = obtain_interpolated_velocity_over_τ(X_grid, traj_in, vels_in, rhos_in, t, N_f)
    return Vf
end

method = m_nns[3];
# for method in m_tot
    println("Running ", method)
    accl, traj, vels, rhos = load_data_t20(method, T_pred, T_train)
    Vf_pred = obtain_vel_mt_field_values(T_pred, traj, vels, rhos)
    path = "./learned_generalization_mt_rel/vf_pred_Tp$(T_pred)_Tt$(T_train)_$(method)_Mt$(mt).npy"
    npzwrite(path, Vf_pred)
# end
