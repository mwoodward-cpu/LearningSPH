using NPZ, Plots
using LaTeXStrings, SpecialFunctions
using BSON: @load
using Flux
using ForwardDiff
using Statistics

include("color_scheme_utils.jl")

function simulate_eosgt(Pnn, p_gt, ρ_gt, sim_time=15)
    gr(size=(1000,800))
		sim_path = "./learned_sims/eos_gt_t.mp4"

    println("**************** Simulating pressure ***************")
    #theme(:juno)
    anim = @animate for i ∈ 1:500
			println("time step = ", i)
			scatter(ρ_gt[i,:], p_gt[i,:])
			plot!(ρ -> Pnn(ρ), 0.9, 1.1, label=L"P_{\theta}", linewidth = 2.5)
		end
    gif(anim, sim_path, fps = round(Int, 500/sim_time))
    println("****************  Simulation COMPLETE  *************")
end


function plot_eos(P_nn, P, cab, gab, c_p, g_p, ρ_gt, P_gt, po_cub, po_wab, po_w2ab, c_w2ab, g_w2ab)
  gr(size=(700,600))
  path = "./learned_figures/comp_eos_learned_update_eos.png"
  po = 0.749; xe = 1.1; xs = 0.9
  # plt = plot(ρ -> P_nn(ρ), 0.9, 1.1, label=L"P_{NN_{\theta}}", linewidth = 2.7)
  plt = plot(ρ -> P(ρ, cab, gab, 0.485), xs, xe, label=L"P(\rho): {W(a,b)}", color=cs[8], markershape=:xcross, ms = 5.5, linestyle=:dashdot, linewidth = 3.4, legendfontsize=16, legend=:bottomright)
  plot!(ρ -> P(ρ, c_w2ab, g_w2ab, 0.465), xs, xe, label=L"P(\rho): {W_2(a,b)}", color=cs[9], markershape=:auto, ms = 5.5, linestyle=:dashdot, linewidth = 3.4, legendfontsize=16, legend=:bottomright)
  plot!(ρ -> P(ρ, c_p, g_p, 0.41), xs, xe, label=L"P(\rho): W_{{cub}}", linewidth = 3.4, legendfontsize=16, color=cs[6], legend=:bottomright)
  plot!(ρ -> Pnn(ρ) + po - 0.039, xs, xe, label=L"P(\rho): {NN_\theta}", markershape=:dtriangle, ms = 5.5, linestyle=:dash, linewidth = 3.4, legendfontsize=16, color = cs[5], legend=:bottomright)
  # plot!(ρ -> P_dns(ρ, 0.845, 1.4), xs, xe, label=L"P_{gt}(\rho)(c_{ab})", linewidth = 2.5, color = "black")
  scatter!(ρ_gt[20,:], p_gt[20,:], label=L"P_{gt}", markershape=:+, ms = 4.2, color="black");
  title!(L"\textrm{Comparing ~ Learned ~ EoS: } M_t = 0.08", titlefont=20)
  xlabel!(L"\rho", xtickfontsize=12, xguidefontsize=18)
  ylabel!(L"P(\rho)", ytickfontsize=12, yguidefontsize=18)
  savefig(plt, path)
  display(plt)
end


function compute_c(Pres, c, g, po)
	dp_dρ(x) = ForwardDiff.derivative(x -> Pres(x, c, g, po), x)
	c_1 = sqrt(dp_dρ(1))
	c_mean = mean(sqrt.(dp_dρ.(0.9:0.01:1.1)))
	return c_1, c_mean
end

function compute_c_nn(Pres)
	dp_dρ(x) = ForwardDiff.derivative(x -> Pres(x), x)
	c_1 = sqrt(dp_dρ(1))
	c_mean = mean(sqrt.(dp_dρ.(0.9:0.01:1.1)))
	return c_1, c_mean
end


function compute_emerical_c(p_gt, ρ_gt)
	c_e = zeros(4096); γ = 5/3; γ_1 = 1.4;
	for i in 1:4096
		c_e[i] = sqrt(γ_1 * (p_gt[i]/ρ_gt[i]))
	end
	c_mean = mean(c_e)
	return c_mean
end

function P(rho, c, g, po)
  return c^2 * (rho^g) + po;
end

function P_dns(rho, c, g)
  return c^2 * (rho^g);
end


include("./load_models_t20.jl")
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()

# include("./load_models_t50.jl")
# d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t50()

p_w2ab = npzread("$(d1)/params_intermediate.npy")
p_wab = npzread("$(d2)/params_intermediate.npy")
p_wcub = npzread("$(d4)/params_intermediate.npy")

c_w2ab = p_w2ab[1]; g_w2ab = p_w2ab[4]; po_w2ab = p_w2ab[5];
c_wab = p_wab[1]; g_wab = p_wab[4]; po_wab = p_wab[5];
c_cub = p_wcub[1]; g_cub = p_wcub[4]; po_cub = p_wcub[5];

println("bgp_wab = ", po_wab)
println("bgp_cub = ", po_cub)

gt_data_dir = "./equil_ic_data/mt008/"
p_gt = npzread("$(gt_data_dir)/p_traj_4k_unif.npy")
ρ_gt = npzread("$(gt_data_dir)/rho_traj_4k_unif.npy")


p_fin = npzread("$(d8)/params_intermediate.npy")
@load "$(d8)/NN_model.bson" NN
p_hat, re = Flux.destructure(NN)   #flatten nn params
Pnn(ρ) = re(p_fin[1:end-3])([ρ])[1];


plot_eos(Pnn, P, c_wab, g_wab, c_cub, g_cub, ρ_gt, p_gt, po_wab, po_cub, po_w2ab, c_w2ab, g_w2ab)


cs_w2ab = compute_c(P, c_w2ab, g_w2ab, po_w2ab)
cs_wab = compute_c(P, c_wab, g_wab, po_wab)
cs_cub = compute_c(P, c_cub, g_cub, po_cub)
cs_nn = compute_c_nn(Pnn)
cs_gt = compute_emerical_c(p_gt, ρ_gt)

println("cs_w2ab = ", cs_w2ab)
println("cs_wab = ", cs_wab)
println("cs_cub = ", cs_cub)
println("cs_nn = ", cs_nn)
println("cs_gt = ", cs_gt)


function plot_gt_eos()
  gr(size=(700,600))
  path = "./learned_figures/comp_gt_eos_learned.png"
  p_o = 0.713; xe = 1.08; xs = 0.92
  # plt = plot(ρ -> P_nn(ρ), 0.9, 1.1, label=L"P_{NN_{\theta}}", linewidth = 2.7)
  plt = plot(ρ -> P_dns(ρ, 0.845, 5/3), xs, xe, label=L"c = 0.845, \gamma = 5/3", linewidth = 2.5, color = "blue")
  plot!(ρ -> P_dns(ρ, 0.845, 1.4), xs, xe, label=L"c = 0.845, \gamma = 1.4", linewidth = 2.5, color = "black")
  scatter!(ρ_gt[20,:], p_gt[20,:], label=L"P_{gt}", markershape=:+, color="grey");
  title!(L"P_{gt}(\rho) \approx c^2 \rho^\gamma \textrm{ Comparing EoS: } M_t = 0.08", titlefont=20)
  xlabel!(L"\rho", xtickfontsize=10, xguidefontsize=18)
  ylabel!(L"P(\rho)", ytickfontsize=10, yguidefontsize=18)
  savefig(plt, path)
  display(plt)
end

# plot_gt_eos()


#
# # simulate_eosgt(Pnn, p_gt, ρ_gt)
# println("  *****************************  ")
#
# c_pab, cm_pab = compute_c(rho -> P(rho, c_ab, g_ab, 0.713))
# println("Wab: c = ", c_pab, "   Wab: cm = ", cm_pab)
#
# c_pcub, cm_pcub = compute_c(rho -> P(rho, c_phy, g_phy, 0.713))
# println("Wcub: c = ", c_pcub, "   Wcub: cm = ", cm_pcub)
#
# c_pnn, cm_pnn = compute_c(rho -> Pnn(rho))
# println("Pnn: c = ", c_pnn, "   Pnn: cm = ", cm_pnn)
