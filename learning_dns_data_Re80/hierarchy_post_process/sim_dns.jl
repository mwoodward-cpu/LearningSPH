using Plots, LaTeXStrings, NPZ


function sim_dns(pos, t_end, sim_time=14)
    gr(size=(600,600))
    sim_path = "./traj_dns.mov"
    println("**************** Simulating the particle flow ***************")
    n_2 = round(Int,N/2); m_s = 5.0
    anim = @animate for i ∈ 1 : t_end
         println("sim time = ", i)
         Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
         title = L"\textrm{DNS: Lagrangian ~ tracers}", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms = m_s)
         Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red", ms = m_s)
    end
    mov(anim, sim_path, fps = round(Int, t_end/sim_time))
    println("****************  Simulation COMPLETE  *************")
end

traj = npzread("./equil_ic_data/mt016/pos_traj_4k_unif.npy")
t_end, N, D = size(traj)
sim_dns(traj, t_end)
