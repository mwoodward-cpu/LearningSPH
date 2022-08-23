"""
Computing Energy Spectrum for SPH particle flow data.

  1) First create a grid (X_grid) of size N,D
  2) Use SPH smoothing for interpolating scattered data onto X_grid to get V_int
  3) Organize V_int to be on periodic physical grid to feed into FFT
  4) Compute fft of interpolated veclocity field
  5) Compute E_K_vec = 0.5*(u_K^2 + v_K^2 + w_K^2) where K is wavevector
  6) Compute the spherical integral of E_K_vec to get E_k where k = |K|

"""

using FFTW, NPZ, LaTeXStrings

# m_runs = "phys_inf"
# m_runs = "nns"
# m_runs = "comb"


# t_s = 6*70

m_phys = ["phys_inf_theta_po_liv_Pi", "phys_inf_Wliu_theta_po_liv_Pi",
		"phys_inf_Wab_theta_po_liv_Pi", "phys_inf_W2ab_theta_po_liv_Pi"];
#

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "grad_p_theta_alpha_beta_liv_Pi", "eos_nn_theta_alpha_beta_liv_Pi"];
# m_tot = vcat(m_phys, m_nn);
m_comb = vcat(m_nns, m_phys)

methods_phys = [L"W_{cub}" L"W_{quart}" L"W(a,b)" L"W_{2}(a,b)"];

#
methods_nn = [L"NODE" L"\sum NN" L"Rot-Inv" L"(\nabla P)_{nn}" L"P_{nn}"] # L"W_{ab; p_0, \theta}"];
methods_comb = [L"NODE" L"\sum NN" L"RotInv" L"(\nabla P)_{nn}" L"P_{nn}" L"W_{cub}" L"W_{quart}" L"W(a,b)" L"W_{2}(a,b)"] # L"W_{ab; p_0, \theta}"];

l_method = "lf"


#colors of plots
include("./color_scheme_utils.jl")
cs1=[c1,c2,c3,c4,c5,c6,c7,c8,c9];
cs2=[c6,c7,c8,c9];
cs3=[c1,c2,c3,c4,c5];

if m_runs=="phys_inf" cs = cs2 end
if m_runs=="comb" cs = cs1 end


#----------Load data
t_start = 1;
θ = 0.0002;
h = 0.335;
t_coarse = 1
dt = t_coarse*0.04;


include("./data_loader.jl")
pos_path = "./equil_ic_data/mt008/pos_traj_4k_unif.npy"
vel_path = "./equil_ic_data/mt008/vel_traj_4k_unif.npy"
rho_path = "./equil_ic_data/mt008/rho_traj_4k_unif.npy"
traj_gt, vels_gt, rhos_gt = load_dns_tracers(pos_path, vel_path, rho_path);

traj_gt = traj_gt[t_start:t_coarse:end, :, :]
vels_gt = vels_gt[t_start:t_coarse:end, :, :]
rhos_gt = rhos_gt[t_start:t_coarse:end, :]

D = size(traj_gt)[3];
N = size(traj_gt)[2]; N_f = N;
m = (2.0 * pi)^D / N;



#----------GRID

nz = ny = nx = round(Int,(N)^(1/3))
L = Lx = Ly = 2*pi


function obtain_unif_mesh_from_tracers(traj)
	X_grid = traj[1,:,:]
	return X_grid
end
X_grid = obtain_unif_mesh_from_tracers(traj_gt)

#smoothing kernel
function W(r, h)
  sigma = 1/(pi*h^3)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end


n_hash = floor(Int, 2*pi / h);   l_hash = 2*pi / n_hash;
function obtain_interpolated_velocity(X_grid, X, V, rho, N_f)
  Vf = zeros(N,D);
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash, k in 1 : n_hash];
  for n in 1 : N_f
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1,
               floor(Int, X[n, 3] / l_hash) + 1], n);
  end

  XX = zeros(D);
  for n in 1 : N
    x_hash = [floor(Int, X_grid[n, 1] / l_hash) + 1,
  			floor(Int, X_grid[n, 2] / l_hash) + 1,
  			floor(Int, X_grid[n, 3] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
  	xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
  	while (xb_hash > n_hash)    xb_hash -= n_hash;   end
  	for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
  	  yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
  	  while (yb_hash > n_hash)    yb_hash -= n_hash;   end
  	  for za_hash in x_hash[3] - 2 : x_hash[3] + 2
  		zb_hash = za_hash;    while (zb_hash < 1)    zb_hash += n_hash;   end
  		while (zb_hash > n_hash)    zb_hash -= n_hash;   end
  		for n2 in hash[xb_hash, yb_hash, zb_hash]
          close = true; r2 = 0.;
          for i in 1 : D
            XX[i] = X_grid[n, i] - X[n2, i];
            while (XX[i] > pi)   XX[i] -= 2. * pi;   end
            while (XX[i] < -pi)   XX[i] += 2. * pi;   end
            r2 += XX[i] * XX[i];
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)

			for i in 1 : D
	            Vf[n, i] += m * W(sqrt(r2), h) * V[n2, i] / rho[n2];
			end
          end
        end
    end   end   end
  end
  Vf = reshape(Vf, (nx, ny, nz, D))
  return Vf
end


#---------------- Compute energy spectrum

function obtain_energy_spectrum(V_int)
    V_hat = fft(V_int)
    ampl_u = abs.(V_hat[:,:,:,1]) / nx
    ampl_v = abs.(V_hat[:,:,:,2]) / ny
	ampl_w = abs.(V_hat[:,:,:,3]) / nz
    EK_u = fftshift(ampl_u.^2)
    EK_v = fftshift(ampl_v.^2)
	EK_w = fftshift(ampl_w.^2)

    k_radius = round(Int, (ceil((sqrt((nx)^2+(ny)^2+(nz)^2))/2.0)+1))
    centerx = round(Int, (nx/2)) + 1
    centery = round(Int, (ny/2)) + 1
	centerz = round(Int, (nz/2)) + 1
    Ek_u_avsphr = zeros(k_radius) ## U comp average over sphere
    Ek_v_avsphr = zeros(k_radius)
	Ek_w_avsphr = zeros(k_radius)
	Ek_avsphr = zeros(k_radius) #Total spectral energy average over sphere

    for i in 1 : nx
			for j in 1 : ny
				for k in 1 : nz
					wn =  round(Int, (sqrt((i-centerx)^2+(j-centery)^2+(k-centerz)^2))) + 1
					Ek_u_avsphr[wn] = Ek_u_avsphr[wn] + EK_u[i,j,k]
					Ek_v_avsphr[wn] = Ek_v_avsphr[wn] + EK_u[i,j,k]
					Ek_w_avsphr[wn] = Ek_w_avsphr[wn] + EK_w[i,j,k]
				end
		end
	end

	Ek_avsphr = 0.5 * (Ek_u_avsphr + Ek_v_avsphr + Ek_w_avsphr)
	k_wavenumber = 0:(length(Ek_avsphr) - 1)
	return k_wavenumber, Ek_avsphr
end



using Plots

function plot_energy_spectrum(Ek, k_wave, t)
	gr(size=(700,600))
	println("*************** generating plots ******************")
	Ek_path = "./learned_figures/energy_spectrum_t$(t).png"
	plt = plot(k_wave[2:end], Ek[2:end] .+ 1e-5, label = "t = $(t)",
	 			linewidth = 2.5, yaxis =:log)

		title!(L"\textrm{Energy Spectrum} ", titlefont=20)
		xlabel!(L"k", xtickfontsize=14, xguidefontsize=20)
		ylabel!(L"E_k", ytickfontsize=14, yguidefontsize=20)
	# k_ = 5:45
	# k_2 = 2:18
	# scale_5_3 = (2* maximum(Ek_t)) * k_2 .^(-5/3)
	# scale_3 = (2* 1e2 * maximum(Ek_t)) * k_ .^(-3)
	# plt_ek5 = plot!(k_2, scale_5_3, label = "-5/3   K41")
	# plt_Ek6 = plot!(k_, scale_3, label = "-3      K41")
	display(plt)
	savefig(plt, Ek_path)
end



#---------Testing functions


function load_gtvel_field(t_s)
	data_dir_phy = "./learned_field_data"
	vel_path = "$(data_dir_phy)/Vf_gt_Tp500.npy"
	vels = npzread(vel_path); t_dim = size(vels)[1];
	Vf = reshape(vels, (t_dim, nx, ny, nz, D));
	return Vf[t_s, :, :, :, :];
end

Vf_gt = load_gtvel_field(t_s)
k_, Ek_gt = obtain_energy_spectrum(Vf_gt);

function load_vel_field(method)
	data_dir_phy = "./learned_field_data"
	vel_path = "$(data_dir_phy)/Vf_pr_$(method)_lf_Tp500_Tt20.npy"
	vels = npzread(vel_path); t_dim = size(vels)[1];
	Vf = reshape(vels, (t_dim, nx, ny, nz, D))
	return Vf
end

function obtain_all_fields(all_methods, t_s)
    n_methods = size(all_methods)[1];
    Vf_all_t = zeros(nx, ny, nz, D, n_methods);
    ii = 1;
    for m_ in all_methods
        Vf_all_t[:,:,:,:, ii] = load_vel_field(m_)[t_s,:,:,:,:]
        ii += 1;
    end
    return Vf_all_t
end


function obtain_all_spectrum(all_methods, Vf_all_t)
	data_dir_phy = "./learned_field_data"
	vel_path = "$(data_dir_phy)/Vf_gt_Tp500.npy"
	vels = npzread(vel_path); t_dim = size(vels)[1];
	Vf = reshape(vels, (t_dim, nx, ny, nz, D))
	k_wave, Ek_wcub = obtain_energy_spectrum(Vf[1,:,:,:,:]);
	n_methods = size(all_methods)[1];
	n_k = size(k_wave)[1]
	Ek_all = zeros(n_k, n_methods); k_all = zeros(n_k, n_methods);
	for i in 1 : n_methods
		k_all[:, i], Ek_all[:, i] = obtain_energy_spectrum(Vf_all_t[:,:,:,:, i])
	end
	return k_wave, Ek_all
end



function plot_energy_spectrum_phys(k_wave, Ek)
	gr(size=(900,750))
	t_at = round(t_s*dt, digits=2)
	println("*************** generating plots ******************")
	Ek_path = "./learned_figures/energy_spectrum_t$(t_s)_$(m_runs).png"
	if m_runs=="phys_inf" methods = methods_phys; end
	if m_runs=="nns" methods = methods_nn; end
	if m_runs=="comb" methods = methods_comb; end

	# lw_ = 8.0; ms_ = 9.0
	plt = plot(k_wave[2:end], Ek[2:end,1] .+ 1e-7, yaxis=:log, linewidth = lw, color=cs[1],
				label = methods[1], markershape = :heptagon, ms = ms_, legendfontsize=16, legend=:bottomleft)

	plot!(k_wave[2:end], Ek[2:end,2] .+ 1e-7, yaxis=:log, linewidth = lw, color=cs[2];
				label = methods[2] , legendfontsize=16, markershape = :star8, ms = ms_)
	#
	plot!(k_wave[2:end], Ek[2:end,3] .+ 1e-7, yaxis=:log, linewidth = lw,
				label = methods[3] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[3])
	#
	plot!(k_wave[2:end], Ek[2:end,4] .+ 1e-7, yaxis=:log, linewidth = lw,
				label = methods[4] , legendfontsize=16, markershape =:rect, ms = ms_, color = cs[4])


	plot!(k_wave[2:end], Ek_gt[2:end,:] .+ 1e-7, yaxis=:log, linewidth = lw,
							label = L"\textrm{Truth}" , legendfontsize=legend_fs, color = "black")


		title!(L"\textrm{Energy ~ Spectrum: } t = %$t_at (s)", titlefont=title_fs)
		xlabel!(L"k", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
		ylabel!(L"E_k", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

	display(plt)
	savefig(plt, Ek_path)
end



function plot_energy_spectrum_comb(k_wave, Ek)
	# gr(size=(700,600))
	t_at = round(t_s*dt, digits=2)
	println("*************** generating plots ******************")
	Ek_path = "./learned_figures/energy_spectrum_t$(t_s)_$(m_runs).png"
	if m_runs=="phys_inf" methods = methods_phys; end
	if m_runs=="nns" methods = methods_nn; end
	if m_runs=="comb" methods = methods_comb; end

		# lw_ = 8.0; ms_ = 9.0
		plt = plot(k_wave[2:end], Ek[2:end,1] .+ 1e-7, yaxis=:log, linewidth = lw, color=cs[1],
					label = methods[1], markershape = :heptagon, ms = ms_, legendfontsize=16, legend=:bottomleft)

		plot!(k_wave[2:end], Ek[2:end,2] .+ 1e-7, yaxis=:log, linewidth = lw, color=cs[2];
					label = methods[2] , legendfontsize=16, markershape = :star8, ms = ms_)
		#
		plot!(k_wave[2:end], Ek[2:end,3] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[3] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[3])
		#
		plot!(k_wave[2:end], Ek[2:end,4] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[4] , legendfontsize=16, markershape =:rect, ms = ms_, color = cs[4])
		#
		plot!(k_wave[2:end], Ek[2:end,5] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[5] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[5])
		#
		plot!(k_wave[2:end], Ek[2:end,6] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[6] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[6])
		#
		plot!(k_wave[2:end], Ek[2:end,7] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[7] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[7])
		#
		plot!(k_wave[2:end], Ek[2:end,8] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[8] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[8])
		#
		plot!(k_wave[2:end], Ek[2:end,9] .+ 1e-7, yaxis=:log, linewidth = lw,
					label = methods[9] , markershape = :circle, ms = ms_, legendfontsize=16, color = cs[9])
		#


		plot!(k_wave[2:end], Ek_gt[2:end,:] .+ 1e-7, yaxis=:log, linewidth = lw,
								label = L"\textrm{Truth}" , legendfontsize=legend_fs, color = "black")


			title!(L"\textrm{Energy ~ Spectrum: } t = %$t_at (s)", titlefont=title_fs)
			xlabel!(L"k", xtickfontsize=tick_fs, xguidefontsize=yaxis_fs)
			ylabel!(L"E_k", ytickfontsize=tick_fs, yguidefontsize=yaxis_fs)

	display(plt)
	savefig(plt, Ek_path)
end




if m_runs=="phys_inf"
	Vf_all_t = obtain_all_fields(m_phys, t_s);
	k_wave, Ek_all = obtain_all_spectrum(m_phys, Vf_all_t)
	plot_energy_spectrum_phys(k_wave, Ek_all)
end

if m_runs=="nns"
	Vf_all_t = obtain_all_fields(m_nns, t_s);
	k_wave, Ek_all = obtain_all_spectrum(m_nns, Vf_all_t)
end

if m_runs=="comb"
	Vf_all_t = obtain_all_fields(m_comb, t_s);
	k_wave, Ek_all = obtain_all_spectrum(m_comb, Vf_all_t)
	plot_energy_spectrum_comb(k_wave, Ek_all)
end





"""
==========================================
Simulate Energy Specturm
==========================================
"""


# function load_vel_field(method)
# 	data_dir_phy = "./learned_fields"
# 	vel_path = "$(data_dir_phy)/vel_field_$(method)_t150.npy"
# 	vels = npzread(vel_path);
# 	return vels
# end
#
# Vint_wcub_t = load_vel_field("phys_inf")
# Vint_wcub_θ_t = load_vel_field("phys_inf_theta")
# Vint_wliu_t = load_vel_field("phys_inf_Wliu")
# Vint_wab_t = load_vel_field("phys_inf_Wab")
# Vint_w2ab_t = load_vel_field("phys_inf_W2ab")
# Vint_wabp_t = load_vel_field("phys_inf_Wab_po")
# Vint_w2abp_t = load_vel_field("phys_inf_W2ab_po")
