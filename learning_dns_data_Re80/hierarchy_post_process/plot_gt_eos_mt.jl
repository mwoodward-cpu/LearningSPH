using Plots, NPZ



gt_data_dir = "./equil_ic_data/mt016"
p_gt16 = npzread("$(gt_data_dir)/p_traj_4k_unif.npy")
ρ_gt16 = npzread("$(gt_data_dir)/rho_traj_4k_unif.npy")


gt_data_dir2 = "./equil_ic_data/mt008"
p_gt8 = npzread("$(gt_data_dir2)/p_traj_4k_unif.npy")
ρ_gt8 = npzread("$(gt_data_dir2)/rho_traj_4k_unif.npy")


gt_data_dir3 = "./equil_ic_data/mt004"
p_gt4 = npzread("$(gt_data_dir3)/p_traj_4k_unif.npy")
ρ_gt4 = npzread("$(gt_data_dir3)/rho_traj_4k_unif.npy")




function plot_eos_gt(ρ_gt, p_gt, mt)
    plt = scatter(ρ_gt[20,:], p_gt[20,:], label=L"P_{gt}", markershape=:+, ms = 4.2, color="black");
    title!(L"\textrm{Comparing ~ Learned ~ EoS: } M_t = %$mt", titlefont=20)
    xlabel!(L"\rho", xtickfontsize=12, xguidefontsize=18)
    ylabel!(L"P(\rho)", ytickfontsize=12, yguidefontsize=18)
    display(plt)
end

plot_eos_gt(ρ_gt16, p_gt16, 0.16)
plot_eos_gt(ρ_gt16, p_gt16, 0.08)
plot_eos_gt(ρ_gt16, p_gt16, 0.04)
