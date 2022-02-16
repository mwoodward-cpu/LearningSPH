using Plots, NPZ

dir_Wcub_po = "output_data_unif_tracers_forward_phys_inf_theta_po_lf_itr2000_lr0.01_T30_θ0.00430899795067121_h0.335_tcoarse2_dt0.04_height6_mphys_inf_theta_po_llf_klswitch0"
dir_Wab_po = "output_data_unif_tracers_forward_phys_inf_Wab_po_theta_lf_itr2000_lr0.01_T30_θ0.00430899795067121_h0.335_tcoarse2_dt0.04_height6_mphys_inf_Wab_po_theta_llf_klswitch0"
dir_Wliu = "output_data_unif_tracers_forward_phys_inf_Wliu_theta_kl_lf_itr2000_lr0.01_T30_θ0.00430899795067121_h0.335_tcoarse2_dt0.04_height6_mphys_inf_Wliu_theta_lkl_lf_klswitch0"
dir_W2ab = "output_data_unif_tracers_forward_phys_inf_W2ab_theta_lf_itr2000_lr0.01_T30_θ0.00430899795067121_h0.335_tcoarse2_dt0.04_height6_mphys_inf_W2ab_theta_llf_klswitch0"

p_fin_Wcub = npzread("./learned_models/$(dir_Wcub_po)/params_fin.npy")
p_fin_Wabp = npzread("./learned_models/$(dir_Wab_po)/params_fin.npy")
p_fin_Wliu = npzread("./learned_models/$(dir_Wliu)/params_fin.npy")
p_fin_W2ab = npzread("./learned_models/$(dir_W2ab)/params_fin.npy")


include("./smoothing_w.jl")

h = 0.335; D = 3;
a1 = p_fin_Wabp[5]; b1 = p_fin_Wabp[6];
a2 = p_fin_W2ab[5]; b2 = p_fin_W2ab[6];

# plot_kernels(a1, b1, a2, b2, h)
# plot_kernels_der(a1, b1, a2, b2, h)
# plot_kernels_H(a1, b1, a2, b2, h)

function combined_plot()
	gr(size=(1300,600))
	p1 = plot_kernels(a1, b1, a2, b2, h)
	p2 = plot_kernels_der(a1, b1, a2, b2, h)
	p3 = plot_kernels_H(a1, b1, a2, b2, h)
	plt = plot(p1, p2, p3)
	savefig(plt, "./learned_figures/comparing_ws_combined.png")
	display(plt);
end
combined_plot()



#
# # p_fin_phys = npzread("./learned_models/output_data_unif_tracers_forward_phys_inf_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height4_mphys_inf_lkl_lf_klswitch0/params_fin.npy")
# # p_fin_Wliu = npzread("./learned_models/output_data_unif_tracers_forward_phys_inf_Wliu_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height5_mphys_inf_Wliu_lkl_lf_klswitch0/params_fin.npy")
# a1_i = npzread("./learned_models/output_data_unif_tracers_forward_phys_inf_Wab_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height6_mphys_inf_Wab_lkl_lf_klswitch0/a_out.npy")
# b1_i = npzread("./learned_models/output_data_unif_tracers_forward_phys_inf_Wab_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height6_mphys_inf_Wab_lkl_lf_klswitch0/b_out.npy")
#
# a2_i = npzread("./learned_models/output_data_unif_tracers_forward_phys_inf_W2ab_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height7_mphys_inf_W2ab_lkl_lf_klswitch0/a_out.npy")
# b2_i = npzread("./learned_models/output_data_unif_tracers_forward_phys_inf_W2ab_kl_lf_itr2105_lr0.04_T20_θ0.0002_h0.335_tcoarse2_dt0.04_height7_mphys_inf_W2ab_lkl_lf_klswitch0/b_out.npy")
#
#
#
# function simulate_learned_kernels(h, sim_time=18)
# 	sim_path = "./learned_sims/smoothing_kernels.mp4"
# 	gr(size=(600,600))
# 	l = 3.2
# 	println("**************** Simulating the particle flow ***************")
# 	anim = @animate for i ∈ 1:size(a1_i)[1]
# 		println("time step = ", i)
# 		plt = plot(r -> W_cub(r, h), 0, 2*h, color="indigo", label=L"W_{cubic}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> W_ab(r, h, a1_i[i], b1_i[i]), 0, 2*h, color="forestgreen", linestyle=:dash, label=L"W_{ab}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> W_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"W_{Liu}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> W2_ab(r, 2*h, a2_i[i], b2_i[i]), 0, 2*h, color="purple", linestyle=:dot, label=L"W2_{ab}", linewidth = l, legendfontsize=12)
#
# 	    title!(L"\textrm{Comparing: } W_{\theta}", titlefont=16)
# 	    xlabel!(L"r", xtickfontsize=10, xguidefontsize=16)
# 	    ylabel!(L"W(r, h)", ytickfontsize=10, yguidefontsize=14)
# 	end
# 	gif(anim, sim_path, fps = round(Int, size(a1_i)[1]/sim_time))
# 	println("****************  Simulation COMPLETE  *************")
# end
#
#
# function simulate_learned_Dr_kernels(h, sim_time=18)
# 	sim_path = "./learned_sims/smoothing_kernels_dr.mp4"
# 	gr(size=(600,600))
# 	l = 3.2
# 	println("**************** Simulating the particle flow ***************")
# 	anim = @animate for i ∈ 1:size(a1_i)[1]
# 		println("time step = ", i)
# 		plt = plot(r -> ∂rW_cub(r, h), 0, 2*h, color="indigo", label=L"\partial_r W_{cubic}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> ∂rW_ab(r, h, a1, b1), 0, 2*h, color="forestgreen", linestyle=:dash, label=L"\partial_r W_{ab}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> ∂rW_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"\partial_r W_{Liu}",linewidth = l, legendfontsize=12)
# 	    plot!(r -> ∂rW2_ab(r, 2*h, a2, b2), 0, 2*h, color="purple", linestyle=:dot, label=L"\partial_r W2_{ab}",linewidth = l, legendfontsize=12)
#
# 	    title!(L"\textrm{Comparing: } \partial_r W_{\theta}", titlefont=16)
# 	    xlabel!(L"r", xtickfontsize=10, xguidefontsize=16)
# 	    ylabel!(L"\partial_r W_{\theta}(r, h)", ytickfontsize=10, yguidefontsize=14)
# 	end
# 	gif(anim, sim_path, fps = round(Int, size(a1_i)[1]/sim_time))
# 	println("****************  Simulation COMPLETE  *************")
# end
#
#
# function simulate_learned_H(h, sim_time=18)
# 	sim_path = "./learned_sims/smoothing_kernels_H.mp4"
# 	gr(size=(600,600))
# 	l = 3.2
# 	println("**************** Simulating the particle flow ***************")
# 	anim = @animate for i ∈ 1:size(a1_i)[1]
# 		println("time step = ", i)
# 		plt = plot(r -> H_cub(r, h), 0, 2*h, color="indigo", label=L"H_{cubic}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> H_ab(r, h, a1, b1), 0, 2*h, color="forestgreen", linestyle=:dash, label=L"H_{ab}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> H_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"H_{Liu}", linewidth = l, legendfontsize=12)
# 	    plot!(r -> H2_ab(r, 2*h, a2, b2), 0, 2*h, color="purple", linestyle=:dot, label=L"H2_{ab}", linewidth = l, legendfontsize=12)
#
# 	    title!(L"\textrm{Comparing: } H_{\theta} = \frac{\partial_r W}{r}", titlefont=16)
# 	    xlabel!(L"r", xtickfontsize=10, xguidefontsize=16)
# 	    ylabel!(L"H(r, h)", ytickfontsize=10, yguidefontsize=14)
# 	end
# 	gif(anim, sim_path, fps = round(Int, size(a1_i)[1]/sim_time))
# 	println("****************  Simulation COMPLETE  *************")
# end
#
#
#
# function simulate_learned_combined(h, sim_time=18)
# 	sim_path = "./learned_sims/smoothing_kernels_combined.mp4"
# 	gr(size=(1300,600))
# 	l = 3.2
# 	println("**************** Simulating the particle flow ***************")
# 	anim = @animate for i ∈ 1:size(a1_i)[1]
# 		println("time step = ", i)
# 		p1 = plot_kernels(a1_i[i], b1_i[i], a2_i[i], b2_i[i], h)
# 		p2 = plot_kernels_der(a1_i[i], b1_i[i], a2_i[i], b2_i[i], h)
# 		p3 = plot_kernels_H(a1_i[i], b1_i[i], a2_i[i], b2_i[i], h)
# 		plt = plot(p1, p2, p3)
# 	end
# 	gif(anim, sim_path, fps = round(Int, size(a1_i)[1]/sim_time))
# 	println("****************  Simulation COMPLETE  *************")
# end
#
# #
# # simulate_learned_kernels(h)
# # simulate_learned_Dr_kernels(h)
# # simulate_learned_H(h)
# # simulate_learned_combined(h)
