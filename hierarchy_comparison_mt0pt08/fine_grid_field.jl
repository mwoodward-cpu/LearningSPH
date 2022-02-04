using NPZ, LaTeXStrings, Plots


l_method = "lf"
T = 20


t_save = 1   #initial time for saving
t_start = 1;
h_kde = 0.9;
r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
n_int = 250;

extern_f = "determistic"
IC = "dns"


θ = 0.0002;
h = 0.165; #n_h = 20, gsp = 5;
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
N = size(traj_gt)[2];
m = (2.0 * pi)^D / N;


function save_output_data(data, path)
    npzwrite(path, data)
end
function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end
make_dir("fine_field_data")

function load_field_data(method)
	data_dir_phy = "./learned_field_data/"
	vf_path = "$(data_dir_phy)/accl_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	vf = npzread(vf_path);
	return vf
end

function load_sim_coarse_data(method)
	data_dir_phy = "./learned_data/"
	tra_path = "$(data_dir_phy)/traj_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	vel_path = "$(data_dir_phy)/vels_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	rho_path = "$(data_dir_phy)/rhos_N4096_T250_h0.335_dns_$(method)_$(l_method).npy"
	traj = npzread(tra_path);
	vels = npzread(vel_path);
	rhos = npzread(rho_path);
	return traj, vels, rhos
end

traj, vels, rhos = load_sim_coarse_data("phys_inf_theta");

include("./interpolate_field.jl")
function obtain_vel_field_values(T, X_grid_fine, gsp)
	Nf = ceil(Int, (2^gsp)^3)
	Vf_pr = obtain_interpolated_vel_field_over_τ(X_grid_fine, traj, vels, rhos, T, Nf)
	return Vf_pr
end

X_grid_fine = obtain_uniform_mesh(D, 5)
Vf_pr = obtain_vel_field_values(T, X_grid_fine, 5)



"""
=======================Check Fields=========================
"""

Nf = ceil(Int, (2^5)^3)
include("./smoothed_contour_js.jl")
function obtain_slice_contour_vel(t, vf_pr, method, gsp)
	x = range(0, 2*pi, length=2^gsp); y = x;
	vp_t = reshape(vf_pr[t,:,:], (2^gsp, 2^gsp, 2^gsp, 3));
	up_t = vp_t[:,:,1,1];
	p1 = contour_js(x, y, up_t, "U_pr", method, t)
end


function obtain_slice_contour_dis_time(n_slices)
	for t in ceil.(Int, range(1, T, length = n_slices))
		println("obtained t = $(t) slice")
		obtain_slice_contour_vel(t, Vf_pr, method, 5)
	end
end


method = "phys_inf_theta"
obtain_slice_contour_dis_time(2)




# println("*************** Obtaining Velocity Field Values ************")
# Nf = ceil(Int, (2^5)^3)
# Vf_pr = obtain_vel_field_values(T, X_grid_fine, Nf)
# println("*************** Velocity Field obtained ************")
#
# make_dir("learned_field_data");
# save_output_data(Vf_gt, "./learned_field_data/Vf_gt_$(method)_$(l_method)_T$(T).npy")
# save_output_data(Vf_pr, "./learned_field_data/Vf_pr_$(method)_$(l_method)_T$(T).npy")
