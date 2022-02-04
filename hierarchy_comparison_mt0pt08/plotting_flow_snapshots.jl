using Plots, NPZ


# method = "phys_inf"; latex_method = "W_{cub}"
# method = "phys_inf_theta"; latex_method = L"W_{cub,\theta}"
# method = "phys_inf_Wliu"; latex_method = "W_{liu}"
# method = "phys_inf_Wab"; latex_method = "W_{ab}"
# method = "phys_inf_Wab_bg_pres"
# method = "phys_inf_Wab_po"; latex_method = "W_{ab,p_0}"
# method = "phys_inf_W2ab"; latex_method = "W_{2ab}"
# method = "phys_inf_W2ab_bgp"
# method = "phys_inf_W2ab_po"; latex_method = "W_{2ab,p_0}"
method = "truth"; latex_method = "Truth"


include("./data_loader.jl")
pos_path = "./wc_dns_4096_unif_longer_time/pos_traj_4k.npy"
vel_path = "./wc_dns_4096_unif_longer_time/vel_traj_4k.npy"
rho_path = "./wc_dns_4096_unif_longer_time/rho_traj_4k.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path)

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;


function load_traj_data(method)
	data_dir_phy = "./learned_data_coarse/"
	tra_path = "$(data_dir_phy)/traj_N4096_T250_ts1_h0.335_dns_θ0.0002_$(method).npy"
	return npzread(tra_path)
end

# traj = load_traj_data(method)
T = 250;

function obtain_snapshots(traj, n_snaps=4)
        m_s = 2.22
        ratio = 1/(n_snaps+2); horz = 1000; vert = ceil(Int, ratio*horz);
        gr(size=(horz,vert))
        # gr(size=(600,600))

        n_2 = round(Int,N/2);
        t_steps = size(traj)[1]; T = t_steps - 1;
        t_range = ceil.(Int, range(20, ceil(Int, T), length=n_snaps));

        p0 = plot(zeros(2,2), xlims = [0, 1], ylims = [0,1], axis=([], false), grid = false)

        p1 =Plots.scatter(traj[1, 1:n_2, 1], traj[1, 1:n_2, 2], traj[1, 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s,
                zlabel=L"%$latex_method", ztickfontsize=1, zguidefontsize=18)
                # zlabel=L"d_t v_i = NN_{\theta}", ztickfontsize=1, zguidefontsize=30)
                Plots.scatter!(traj[1, (n_2+1):end, 1], traj[1, (n_2+1):end, 2],
                traj[1,(n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_0")


        p2 =Plots.scatter(traj[t_range[1], 1:n_2, 1], traj[t_range[1], 1:n_2, 2], traj[t_range[1], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[1], (n_2+1):end, 1], traj[t_range[1], (n_2+1):end, 2],
                traj[t_range[1],(n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_{\lambda}")

        p3 =Plots.scatter(traj[t_range[2], 1:n_2, 1], traj[t_range[2], 1:n_2, 2], traj[t_range[2], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[2], (n_2+1):end, 1], traj[t_range[2], (n_2+1):end, 2],
                traj[t_range[2], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_2")

        p4 =Plots.scatter(traj[t_range[3], 1:n_2, 1], traj[t_range[3], 1:n_2, 2], traj[t_range[3], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[3], (n_2+1):end, 1], traj[t_range[3], (n_2+1):end, 2],
                traj[t_range[3], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"t_3")

        p5 =Plots.scatter(traj[t_range[4], 1:n_2, 1], traj[t_range[4], 1:n_2, 2], traj[t_range[4], 1:n_2, 3],
                xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms=m_s)
                Plots.scatter!(traj[t_range[4], (n_2+1):end, 1], traj[t_range[4], (n_2+1):end, 2],
                 traj[t_range[4], (n_2+1):end, 3], color = "red", ms=m_s)#, title=L"\mathcal{O}(t_{eddy})")


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
        out_path = "./learned_figures/gen_snapshots_$(T)_$(method).png"
        display(plt)
        savefig(plt, out_path);
        # savefig(p1, out_path1); savefig(p2, out_path2); savefig(p3, out_path3); savefig(p4, out_path4); savefig(p5, out_path5);
end

# obtain_snapshots(traj)

if method == "truth"
	obtain_snapshots(traj_gt)
end
