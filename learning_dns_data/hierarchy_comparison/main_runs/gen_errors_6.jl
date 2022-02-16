using Plots, NPZ, LaTeXStrings, Flux
using Flux.Losses, Statistics

function save_output_data(data, path)
    npzwrite(path, data)
end
function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end


m_phys = ["phys_inf_theta", "phys_inf_theta_livescu_ext", "phys_inf_theta_correct_Pi", "phys_inf_Wab_theta",
                        "phys_inf_W2ab_theta", "phys_inf_Wliu_theta", "phys_inf_theta_po", "phys_inf_Wab_po_theta"];

m_nn = ["node_norm", "nnsum2_norm_theta", "grad_p_theta", "eos_nn_theta", "rot_inv"];


method = m_phys[6];
l_method = ARGS[1];
T = parse(Int, ARGS[2]);
lg_method = ARGS[3];

loss_method = l_method;
t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 250;

extern_f = "determistic"
IC = "dns"

θ = 0.0002;
h = 0.335;
t_coarse = 2
dt = t_coarse*0.02;


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


println("*****************    *************")
println("Running Method = ", method)


function load_sim_coarse_data(method)
	data_dir_phy = "./learned_data/"
	acc_path = "$(data_dir_phy)/accl_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	tra_path = "$(data_dir_phy)/traj_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	vel_path = "$(data_dir_phy)/vels_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	rho_path = "$(data_dir_phy)/rhos_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	accl = npzread(acc_path);
	traj = npzread(tra_path);
	vels = npzread(vel_path);
	rhos = npzread(rho_path);
	return accl, traj, vels, rhos
end

accl, traj, vels, rhos = load_sim_coarse_data(method);


include("./loss_functions.jl")
Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(X_grid, traj_gt, vels_gt, rhos_gt, T, N_f)
function obtain_vel_field_values(T)
	Vf_pr,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(X_grid, traj, vels, rhos, T, N_f)
	return Vf_pr
end

println("*************** Obtaining Velocity Field Values ************")
Vf_pr = obtain_vel_field_values(T)
println("*************** Velocity Field obtained ************")

make_dir("learned_field_data");
save_output_data(Vf_gt, "./learned_field_data/Vf_gt_$(method)_$(l_method)_T$(T).npy")
save_output_data(Vf_pr, "./learned_field_data/Vf_pr_$(method)_$(l_method)_T$(T).npy")


"""
==========================================================================
Plotting Velocity field: Qualitative Gen Error
==========================================================================
"""

include("./smoothed_contour_js.jl")
function obtain_slice_contour_vel(t, vf_gt, vf_pr, method, gsp)
	x = range(0, 2*pi, length=2^gsp); y = x;
	vp_t = reshape(vf_pr[t,:,:], (2^gsp, 2^gsp, 2^gsp, 3));
	vg_t = reshape(vf_gt[t,:,:], (2^gsp, 2^gsp, 2^gsp, 3));
	up_t = vp_t[:,:,1,1]; ug_t = vg_t[:,:,1,1];
	# println(up_t)
	p1 = contour_js(x, y, up_t, "U_pr", method, t)
	p2 = contour_js(x, y, ug_t, "U_gt", method, t)
	# p = [p1 p2]
	# path_2 = "./learned_figures/pr_gt_field.png"
	# PlotlyJS.savefig(p, path_2)
	# PlotlyJS.savefig(p1, path); PlotlyJS.savefig(p2, path)
	# return x, y, up_t, ug_t
end


function obtain_slice_contour_dis_time(n_slices)
	for t in ceil.(Int, range(1, T, length = n_slices))
		println("obtained t = $(t) slice")
		obtain_slice_contour_vel(t, Vf_gt, Vf_pr, method, 4)
	end
end

obtain_slice_contour_dis_time(5)



"""
==========================================================================
Obtaining generalization error data: over time
==========================================================================

"""

include("./kde_G.jl")
include("./loss_functions_gen.jl")
make_dir("learned_generalization");

function obtain_vf_t(t)
        Vf = zeros(t+1,N,D);
        Vf_truth = Vf_gt[1:(t+1), :, :];
        Vf_pred = Vf_pr[1:(t+1), :, :];
        return Vf_truth, Vf_pred
end

function compute_avg_kinetic_energy(rhos, vels, t)
    tke = zeros(t);
    for i in 1 : t
        tke[i] = 0.5*mean(rhos[t,:] .* (vels[t, : ,1].^2 .+ vels[t, : ,2].^2 .+ vels[t, : ,3].^2));
    end
    avg_ke = mean(tke);
    return avg_ke;
end



function obtain_gen_loss_t(t_s, t_end, t_num)
        t_range = ceil.(Int, range(t_s, t_end, length=t_num));
        num_t_idx = size(t_range)[1];
        println("num_t_idx  = ", num_t_idx);
        Lt = zeros(num_t_idx); Lgt = zeros(num_t_idx); rot_errt = zeros(num_t_idx);
		Lgkl_t = zeros(num_t_idx); Lf_norm = zeros(num_t_idx);
        ii = 1;
        for t in t_range
                # Vf_gt,d1,d2,d3,d4,d5,d6 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, t)
                Diff_gt, Vel_inc_gt = obtain_gt_dists(traj_gt, vels_gt, t)
                Diff_pred, Vel_inc_pred = obtain_pred_dists(traj, vels, traj_gt[1,:,:], vels_gt[1,:,:], t);
                avg_ke = compute_avg_kinetic_energy(rhos, vels, t);
                Vf_truth, Vf_pred = obtain_vf_t(t)
                Lt[ii] = compute_L_comp(loss_method, Vel_inc_gt, Vel_inc_pred, Vf_pred, Vf_truth, t)
                Lgt[ii] = compute_Lg(lg_method, Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t);
		Lgkl_t[ii] = compute_Lg("kl_t", Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t);
                Lf_norm[ii] = compute_Lg("lf", Vel_inc_gt, Vel_inc_pred, traj, vels, rhos, Vf_pred, Vf_truth, t)/avg_ke;
                println("T = ", t, "   Loss: $(loss_method) = ", Lt[ii], "  $(lg_method) = ", Lgt[ii], "   kl_t = ", Lgkl_t[ii], "  Lf_norm = ", Lf_norm[ii]);
                ii += 1;
        end
        return Lt, Lgt, Lgkl_t, Lf_norm
end


Lt, Lgt, Lgkl_t, Lf_norm = obtain_gen_loss_t(30, T, 20)
save_output_data(Lt, "./learned_generalization/$(l_method)_loss_t_$(method)_T$(T).npy")
save_output_data(Lgt, "./learned_generalization/$(lg_method)_loss_t_$(method)_T$(T).npy")
save_output_data(Lgkl_t, "./learned_generalization/kl_t_loss_t_$(method)_T$(T).npy")
save_output_data(Lf_norm, "./learned_generalization/lf_norm_loss_t_$(method)_T$(T).npy")



function simulate(pos, sim_time=15)
     sim_path = "./learned_sims/traj_N$(N)_T$(T)_$(IC)_$(method).mp4"
     gr(size=(1800,800))
     println("**************** Simulating the particle flow ***************")
     #theme(:juno)
     n_2 = round(Int,N/2)
     anim = @animate for i ∈ 1:T
 		println("time step = ", i)
          plt_pr = Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
          title = "Learned WCSPH_$(method): N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false)
          Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red")
         plt_gt = Plots.scatter(traj_gt[i, 1:n_2, 1], traj_gt[i, 1:n_2, 2], traj_gt[i, 1:n_2, 3],
          title = "Ground Truth Tracer Particles: N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false)
          Plots.scatter!(traj_gt[i, (n_2+1):end, 1], traj_gt[i, (n_2+1):end, 2], traj_gt[i, (n_2+1):end, 3], color = "red")
         Plots.plot(plt_pr, plt_gt)
     end
     gif(anim, sim_path, fps = round(Int, T/sim_time))
     println("****************  Simulation COMPLETE  *************")
end

simulate(traj)




#------IDEA TODO                                                                                                                                                                                                                                          

# function obtain_iso_vorticity(T, Vort, method)                                                                                                                                                                                                                      
#       path = "./learned_figures/plot_vort_isosurface_$(method).png"                                                                                                                                                                                                 
#       Vort_field = 0.5 * (Vort[T,:,1].^2 .+ Vort[T,:,2].^2 .+ Vort[T,:,3].^2)                                                                                                                                                                                       
#       r = 0.2; v = 1.3*r                                                                                                                                                                                                                                            
#       iso_surface_vorticity(Vort_field, r, v, path)                                                                                                                                                                                                                 
# end                                                                                                                                                                                                                                                                 
#                                                                                                                                                                                                                                                                     
# include("gen_vorticity_outs.jl")                                                                                                                                                                                                                                    
# obtain_iso_vorticity(T, vort, method)                                                                                                                                                                                                                               

