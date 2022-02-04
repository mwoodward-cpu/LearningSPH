using NPZ, Statistics

function compute_Mt(vel, t, c)
	Mt = 0.0
	u_fluc = vec(vel[t, :, :]) .- mean(vel[t, :, :])
	Mt = sqrt(mean(u_fluc.^2)) / c
	return Mt
end


t_coarse = 2; t_start  =1;

include("./data_loader.jl")
pos_path = "./wc_dns_4096_unif_longer_time/pos_traj_4k.npy"
vel_path = "./wc_dns_4096_unif_longer_time/vel_traj_4k.npy"
rho_path = "./wc_dns_4096_unif_longer_time/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)
vels_gt = vels_gt[t_start:t_coarse:end, :, :]

function compute_Mt_t(vel, T, c)
	Mt_t = zeros(T)
	for t in 1 : T
		Mt_t[t] = compute_Mt(vel, t, c)
	end
	return Mt_t
end

Mt_t = compute_Mt_t(vels_gt, 250, 0.7)
