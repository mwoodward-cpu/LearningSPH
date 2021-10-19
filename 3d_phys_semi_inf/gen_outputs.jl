

using Plots, NPZ, LaTeXStrings
ENV["GKSwstype"]="100" #set env variable for UAHPC


file_title = "$(IC)_itr$(n_itrs)_lr$(lr)_T$(T)_D$(D)_N$(N)_c$(c_gt)_α$(α)_β$(β)_h$(h)_nb$(nb)_nint$(n_int)_ts$(t_start)_coarse$(coarse_mult)_height$(height)"
data_out_path = "output_data_$(sens_method)_$(method)_$(loss_method)_$(file_title)_klswitch$(switch_kl_lf)"

function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end

make_dir(data_out_path); make_dir("figures"); make_dir("anim")

function save_output_data(data, path)
    npzwrite(path, data)
end

if (method != "phys_inf")
        @save "./$(data_out_path)/NN_model.bson" NN
		println("**************** NN MODEL SAVED **********************")
end


"""

================================= Plotting parameters ==========================

"""


function plot_4g_param()
    gr(size=(600,600))
    println("*************** generating plots ******************")

    c_gt_data = c_gt * ones(size(c_out))
    α_gt_data = α * ones(size(α_out))
    β_gt_data = β * ones(size(β_out))
    g_gt_data = g * ones(size(g_out))
    xs = 1 : vis_rate :	(vis_rate * size(c_out)[1])

    plt = plot(xs, c_out, label=L"\hat{c}", color="green", linewidth = 2.5)
    plot!(xs, c_gt_data, label=L"c", color="green", linestyle=:dash, linewidth = 2.5)

    plot!(xs, α_out, label=L"\hat{\alpha}", color="blue", linewidth = 2.5)
    plot!(xs, α_gt_data, label=L"\alpha", linestyle=:dash, color = "blue", linewidth = 2.5)

    plot!(xs, β_out, label=L"\hat{\beta}", color="purple", linewidth = 2.5)
    plot!(xs, β_gt_data, label=L"\beta", linestyle=:dash, color = "purple", linewidth = 2.5)

    plot!(xs, g_out, label=L"\hat{\gamma}", color="black", linewidth = 2.5)
    plot!(xs, g_gt_data, label=L"\gamma", linestyle=:dash, color = "black", linewidth = 2.5)

    title!(L"\textrm{PIML SPH: N = } %$N \textrm{, time steps = } %$T", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{Parameters}", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    savefig(plt, "./figures/sph_av_fs_$(loss_method)_$(file_title).png")
end

function plot_4θ_param()
    gr(size=(600,600))
    println("*************** generating plots ******************")

    c_gt_data = c_gt * ones(size(c_out))
    α_gt_data = α * ones(size(α_out))
    β_gt_data = β * ones(size(β_out))
    θ_gt_data = θ_gt * ones(size(θ_out))

    plt = plot(c_out, label=L"\hat{c}", color="green", linewidth = 2.25)
    plot!(c_gt_data, label=L"c", color="green", linestyle=:dash, linewidth = 2.25)

    plot!(α_out, label=L"\hat{\alpha}", color="blue", linewidth = 2.25)
    plot!(α_gt_data, label=L"\alpha", linestyle=:dash, color = "blue", linewidth = 2.25)

    plot!(β_out, label=L"\hat{\beta}", color="purple", linewidth = 2.25)
    plot!(β_gt_data, label=L"\beta", linestyle=:dash, color = "purple", linewidth = 2.25)

    plot!(θ_out, label=L"\hat{\gamma}", color="black", linewidth = 2.25)
    plot!(θ_gt_data, label=L"\gamma", linestyle=:dash, color = "black", linewidth = 2.25)


    title!(L"\textrm{PIML SPH: N = } %$N \textrm{, time steps = } %$T", titlefont=17)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=18)
    ylabel!(L"\textrm{Parameters}", ytickfontsize=10, yguidefontsize=18)
    display(plt)
    savefig(plt, "./figures/sph_av_fs_$(file_title).png")
end


function plot_5param()
    gr(size=(600,600))
    println("*************** generating plots ******************")

    c_gt_data = c_gt * ones(size(c_out))
    α_gt_data = α * ones(size(α_out))
    β_gt_data = β * ones(size(β_out))
    g_gt_data = g * ones(size(g_out))
    h_gt_data = h * ones(size(h_out))

    plt = plot(c_out, label=L"\hat{c}", color="green", linewidth = 2.25)
    plot!(c_gt_data, label=L"c", color="green", linestyle=:dash, linewidth = 2.25)

    plot!(α_out, label=L"\hat{\alpha}", color="blue", linewidth = 2.25)
    plot!(α_gt_data, label=L"\alpha", linestyle=:dash, color = "blue", linewidth = 2.25)

    plot!(β_out, label=L"\hat{\beta}", color="purple", linewidth = 2.25)
    plot!(β_gt_data, label=L"\beta", linestyle=:dash, color = "purple", linewidth = 2.25)

    plot!(g_out, label=L"\hat{\gamma}", color="black", linewidth = 2.25)
    plot!(g_gt_data, label=L"\gamma", linestyle=:dash, color = "black", linewidth = 2.25)

    plot!(h_out, label=L"\hat{h}", color="orange", linewidth = 2.25)
    plot!(h_gt_data, label=L"h", linestyle=:dash, color = "orange", linewidth = 2.25)

    title!(L"\textrm{PIML SPH: N = } %$N \textrm{, time steps = } %$T", titlefont=17)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=18)
    ylabel!(L"\textrm{Parameters}", ytickfontsize=10, yguidefontsize=18)

    display(plt)
    savefig(plt, "./figures/sph_av_fs_$(file_title).png")
end


function plot_5theta_param()
    gr(size=(500,500))
    println("*************** generating plots ******************")

    c_gt_data = c_gt * ones(size(c_out))
    α_gt_data = α * ones(size(α_out))
    β_gt_data = β * ones(size(β_out))
    g_gt_data = g * ones(size(g_out))
    θ_gt_data = θ * ones(size(θ_out))

    plt = plot(c_out, label=L"\hat{c}", color="green", linewidth = 2.5)
    plot!(c_gt_data, label=L"c", color="green", linestyle=:dash, linewidth = 2.5)

    plot!(α_out, label=L"\hat{\alpha}", color="blue", linewidth = 2.5)
    plot!(α_gt_data, label=L"\alpha", linestyle=:dash, color = "blue", linewidth = 2.5)

    plot!(β_out, label=L"\hat{\beta}", color="purple", linewidth = 2.5)
    plot!(β_gt_data, label=L"\beta", linestyle=:dash, color = "purple", linewidth = 2.5)

    plot!(g_out, label=L"\hat{\gamma}", color="black", linewidth = 2.5)
    plot!(g_gt_data, label=L"\gamma", linestyle=:dash, color = "black", linewidth = 2.5)

    plot!(θ_out, label=L"\hat{\theta}", color="orange", linewidth = 2.5)
    plot!(θ_gt_data, label=L"\theta", linestyle=:dash, color = "orange", linewidth = 2.5)

    title!(L"\textrm{PIML SPH: N = } %$N \textrm{, time steps = } %$T", titlefont=17)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=18)
    ylabel!(L"\textrm{Parameters}", ytickfontsize=10, yguidefontsize=18)

    display(plt)
    savefig(plt, "./figures/sph_av_fs_$(file_title).png")
end
"""

=========================== Comparing NN Plots =========================================

"""


function compare_eos(p_h)
    gr(size=(600,600))
    Pnn_comp(ρ) = re(p_h)([ρ])[1]
    rho_data = 0.80:0.01:1.20
    P_gt = Pres.(rho_data, c_gt, g)
    P_nn = Pnn_comp.(rho_data)

    plt = plot(rho_data, P_gt, label="P_gt", color="blue", linewidth = 2.5)
    Plots.scatter!(rho_data, P_nn, marker=:x, markersize=4,
		  color="forestgreen", markercolor = :black,
		  label=L"P_{nn(\theta)}(\rho)", linestyle=:dash, linewidth = 2.5)

	title!(L"\textrm{Learning EoS WCSPH}", titlefont=18)
    xlabel!(L"\rho", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"P(\rho)", ytickfontsize=12, yguidefontsize=20)

    display(plt)
	savefig(plt, "./figures/EOS_$(sens_method)_$(loss_method)_$(method)_$(file_title)_height$(height).png")
end

function compare_W(p_h)
    gr(size=(600,600))
    Wnn_comp(r) = W_nn(r, p_h)
    r_data = 0.0:0.01:(2*h)
    W_gt = W.(r_data, h)
    W_nn = Wnn_comp.(r_data)

    plt = plot(r_data, W_gt, label=L"W_{gt}", color="blue", linewidth = 2.5)
	plot!(x->Wnn_comp(x), 0.0, 2*h, marker=:x, markersize=4,
		  color="forestgreen", markercolor = :black,
		  label=L"W_{nn(\theta)}(r)", linestyle=:dash, linewidth = 2.5)

	title!(L"\textrm{Learning Smoothing Kernel} W", titlefont=18)
    xlabel!(L"r", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"W(r, h)", ytickfontsize=12, yguidefontsize=20)

    display(plt)
	savefig(plt, "./figures/Wnn_$(sens_method)_$(loss_method)_$(method)_$(file_title)_height$(height).png")
end


function save_final_compare_gen_eos(p_h)
    gr(size=(500,500))
    Pnn_comp(ρ) = re(p_h)([ρ])[1]
    rho_data = 0.90:0.005:1.10
    P_gt = Pres.(rho_data, c_gt)
    P_nn = Pnn_comp.(rho_data)

    plt = plot(rho_data, P_gt, label="P_gt", color="blue", linewidth = 2)
    Plots.scatter!(rho_data, P_nn, label="P_nn", color="black", linewidth = 2)

    title!("Learning EoS WCSPH N = $(N), T = $(T+1)")
    xlabel!("ρ")
    ylabel!("P(ρ)")

    display(plt)
    savefig(plt, "./figures/$(method)_eos_$(file_title)_height$(height).png")
end

function save_final_compare_generlization_eos(p_h)
#compare True EoS vs Learned EoS from SPH_AV gt data
    #how does it generalize to denistity not seen in gt data
    gr(size=(500,500))
    Pnn_comp(ρ) = re(p_h)([ρ])[1]
    rho_data = 0.80:0.005:1.20
    P_gt = Pres.(rho_data, c_gt)
    P_nn = Pnn_comp.(rho_data)

    plt = plot(rho_data, P_gt, label="P_gt", color="blue", linewidth = 2)
    Plots.scatter!(rho_data, P_nn, label="P_nn", color="black", linewidth = 2)

    title!("Learning EoS WCSPH N = $(N), T = $(T+1)")
    xlabel!("ρ")
    ylabel!("P(ρ)")
    display(plt)
    savefig(plt, "./figures/$(method)_eos_gen_$(file_title)_height$(height).png")
end






"""

=========================== Prob Loss Plots =========================================

"""


function comparing_Gu(G_u, Vel_inc_pred, path, θ_in, width=0.42)
    gr(size=(500,500))
    x_s = -width
    x_e = width
    G_pred(x) = kde(x, Vel_inc_pred[T, :, 1])
    plt = plot(x->G_u(x), x_s, x_e, label=L"G(\tau, d)",
                color="indigo", linewidth = 2.5)
    plot!(x->G_pred(x), x_s, x_e, marker=:x, markersize=4, color="forestgreen",
          markercolor = :black, label=L"G_{\theta}(\tau, d)",
          linestyle=:dash, linewidth = 2.5)

    title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
    xlabel!(L"\textrm{Velocity increment}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"G_u \textrm{ distribution}", ytickfontsize=12, yguidefontsize=20)

    display(plt)
    if path == "train"
        out_path = "./figures/Gtrain_$(loss_method)_$(method)_kde_D$(D)_N$(N)_M$(M)_θ$(θ_in).png"
    elseif path == "test"
        out_path = "./figures/Gtest_$(loss_method)_$(method)_kde_D$(D)_N$(N)_M$(M)_θ$(θ_in).png"
    end
    savefig(plt, out_path)
end




function animate_Gu_fixt(n_itrs, Vel_inc_pred_k, width=0.42, sim_time=10)
    file_out = "./anim/$(method)_$(loss_method)_Gu_$(file_title).mp4"
    x_s = -width
    x_e = width
    Gu_pred(k, x) = kde(x, Vel_inc_pred_k[k, T, :])
    gr(size=(600,600))
    println("**************** Animating ***************")
    anim = @animate for i ∈ 1 : n_itrs
        plt = plot(x->G_u(x), x_s, x_e, label=L"G(\tau, d)",
                    color="indigo", linewidth = 2.5)
        plot!(x->Gu_pred(i, x), x_s, x_e, marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"G_{\theta}(\tau, d)", linestyle=:dash, linewidth = 2.5)
        title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
        xlabel!(L"\textrm{Velocity increment}", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"G_u \textrm{ distribution}", ytickfontsize=12, yguidefontsize=20)
    end

    gif(anim, file_out, fps = ceil(Int, n_itrs/sim_time))
    println("**************** Animation Complete ***************")
end



function animate_learning_EoS(n_itr, rho_data, P_gt, P_nn, sim_time=10)
    file_out = "./anim/eosNN_$(method)_$(file_title)_height$(height).mp4"
    gr(size=(600,600))
    println("**************** Animating ***************")
    anim = @animate for i ∈ 1 : n_itr
        plot(rho_data, P_gt, label=L"P_{gt}", color="blue", linewidth = 2.5)
        Plots.scatter!(rho_data, P_nn[i,:], marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"P_{nn(\theta)}(\rho)", linestyle=:dash, linewidth = 2.5)

        title!(L"\textrm{Learning EoS } P(\rho) ", titlefont=20)
        xlabel!(L" \rho", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"P(\rho)", ytickfontsize=12, yguidefontsize=20)
    end
    gif(anim, file_out, fps = ceil(Int, n_itr/sim_time))
    println("**************** Animation Complete ***************")
end


function animate_learning_W(n_itr, r_data, W_gt, W_nn, sim_time=10)
    file_out = "./anim/Wnn_$(method)_$(file_title)_height$(height).mp4"
    gr(size=(600,600))
    println("**************** Animating ***************")
    anim = @animate for i ∈ 1 : n_itr
        plot(rho_data, P_gt, label=L"W_{gt}", color="blue", linewidth = 2.5)
        Plots.scatter!(rho_data, P_nn[i,:], marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"W_{nn}(\theta)(r)", linestyle=:dash, linewidth = 2.5)

        title!(L"\textrm{Learning Smoothing Kernel} W(r) ", titlefont=20)
        xlabel!(L" r", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"W(r, h)", ytickfontsize=12, yguidefontsize=20)
    end
    gif(anim, file_out, fps = ceil(Int, n_itr/sim_time))
    println("**************** Animation Complete ***************")
end



"""

=========================== Loss Plots =========================================

"""


function plot_loss_itr()
    println(" ===== Plotting Loss =====")
    gr(size=(600,600))
    xs = 1 : vis_rate : (vis_rate * size(L_out)[1])
    plt = plot(xs, L_out, label="$(loss_method)", color="blue", yaxis=:log, linewidth = 2.5)

    title!(L"\textrm{PIML - SPH - Loss: N = } %$N", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{Loss}", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    savefig(plt, "./figures/loss_$(method)_$(loss_method)_$(file_title).png")
end



function plot_rot_itr()
    println(" ===== Plotting rotation error =====")
    gr(size=(600,600))
    xs = 1 : vis_rate : (vis_rate * size(rot_RF)[1])

    plt2 = plot(xs, rot_RF, label="||RF - F(RY)||", color="blue", yaxis=:log, linewidth = 2.25)
    title!(L"\textrm{WCSPH: Rotational - error}", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||RF(Y) - F(RY)||_2", ytickfontsize=10, yguidefontsize=16)

    display(plt2)
    savefig(plt2, "./figures/Rf_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end


function plot_galilean_itr()
    println(" ===== Plotting Translational error =====")
    gr(size=(600,600))
    xs = 1 : vis_rate : (vis_rate * size(galilean_inv)[1])

    plt2 = plot(xs, galilean_inv, label="||F(x-s) - F(x)||", color="blue", yaxis=:log, linewidth = 2.25)
    title!(L"\textrm{WCSPH: Translational - error}", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||F(Y - s) - F(Y)||_2", ytickfontsize=10, yguidefontsize=16)

    display(plt2)
    savefig(plt2, "./figures/Gal_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end
