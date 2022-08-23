#data loader

using NPZ


function load_dns_tracers(pos_path, vel_path, rho_path)
    pos_traj = npzread(pos_path)
    vel_traj = npzread(vel_path)
    rho_traj = npzread(rho_path)
    println("*********** DNS data loaded ****************")
    return pos_traj, vel_traj, rho_traj
end


function periodicize(pos_traj, N)
    for t in 1 : size(pos_traj)[1]
        # pos_traj[t, :, :] = mod.(pos_traj[t, :, :], 2*pi);
        for n in 1 : N
            for i in 1 : D
                while (pos_traj[t, n, i] < 0.)   pos_traj[t, n, i] += 2. * pi;   end
                while (pos_traj[t, n, i] > 2. * pi)   pos_traj[t, n, i] -= 2. * pi;   end
            end
        end
    end
    return pos_traj
end


function coarse_grain_time(traj, vels, rhos, t_coarse, N, D)
	return traj[1:t_coarse:end,:,:], vels[1:t_coarse:end,:,:], rhos[1:t_coarse:end,:]
end


function unif_samp_traj(traj, vels, rhos, N)
	traj = traj[:, rand(1:size(traj)[2], N), :]
	vels = vels[:, rand(1:size(vels)[2], N), :]
	rhos = rhos[:, rand(1:size(rhos)[2], N)]
	return traj, vels, rhos
end



function obtain_ic_unif(gsp, t)
    X_gt = reshape(traj_gt[t,:,:], (2^5, 2^5, 2^5, 3));
    V_gt = reshape(vels_gt[t,:,:], (2^5, 2^5, 2^5, 3));
	rho_gt = reshape(rhos_gt[t,:,:], (2^5, 2^5, 2^5));
    X = zeros(2^gsp, 2^gsp, 2^gsp, 3);
    V = zeros(2^gsp, 2^gsp, 2^gsp, 3);
	rho = zeros(2^gsp, 2^gsp, 2^gsp);
    for i in 1 : 2^gsp
        for j in 1 : 2^gsp
            for k in 1 : 2^gsp
                X[i, j, k, :] = X_gt[2*i, 2*j, 2*k, :];
                V[i, j, k, :] = V_gt[2*i, 2*j, 2*k, :];
				rho[i, j, k] = rho_gt[2*i, 2*j, 2*k];
            end
        end
    end
    X = reshape(X, (2^12, 3));
    V = reshape(V, (2^12, 3));
	rho = reshape(rho, (2^12));
    return X, V, rho
end


function obtain_reduced_traj(T, N_sph, D)
    traj_r = zeros(T, N_sph, D)
	vels_r = zeros(T, N_sph, D)
	rhos_r = zeros(T, N_sph)
    for t in 1 : T
        traj_r[t,:,:], vels_r[t,:,:], rhos_r[t,:] = obtain_ic_unif(gsp, t);
    end
    return traj_r, vels_r, rhos_r
end

# traj_gt, vels_gt, rhos_gt = obtain_reduced_traj(20, N, D)
