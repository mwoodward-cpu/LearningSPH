

using Plots, NPZ, LaTeXStrings
ENV["GKSwstype"]="100" #set env variable for UAHPC


file_title = "$(IC)_itr$(n_itrs)_lr$(lr)_T$(T)_D$(D)_N$(N)_c$(c_gt)_α$(α)_β$(β)_h$(h)_nint$(n_int)_ts$(t_start)_hgt$(height)_coarse$(coarse_mult)"
data_out_path = "output_data_$(sens_method)_$(loss_method)_$(method)_$(file_title)_kl_lf_swt$(switch_kl_lf)"

function make_dir(path)
	if isdir(path) == true
	       println("directory already exists")
	   else mkdir(path)
	end
end

make_dir(data_out_path); make_dir("figures"); make_dir("anim")

if (method != "phys_inf")
	@save "./$(data_out_path)/NN_model.bson" NN
end

function save_output_data(data, path)
    npzwrite(path, data)
end


"""

=========================== Prob Loss Plots =========================================

"""


function comparing_Gu(G_u, Vel_inc_pred, path, θ_in, width=0.42)
    gr(size=(500,500))
    x_s = -width
    x_e = width
    G_pred(x) = kde(x, Vel_inc_pred[T, :, 1])
    plt = plot(x->G_u(x), x_s, x_e, label=L"G(\tau, \delta u)",
                color="indigo", linewidth = 2.5)
    plot!(x->G_pred(x), x_s, x_e, marker=:x, markersize=4, color="forestgreen",
          markercolor = :black, label=L"G_{\theta}(\tau, \delta u)",
          linestyle=:dash, linewidth = 2.5)

    title!(L"\textrm{Comparing - } G(\tau,z) \textrm{ - and - } \hat{G}_{\theta}(\tau,z)", titlefont=18)
    xlabel!(L"\textrm{Velocity - increment}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"G_u \textrm{ - distribution}", ytickfontsize=12, yguidefontsize=20)

    display(plt)
    if path == "train"
        out_path = "./figures/$(sens_method)_$(loss_method)_$(method)_Gtrain_kde_$(file_title)_θ$(θ_in).png"
    elseif path == "test"
        out_path = "./figures/$(sens_method)_$(loss_method)_$(method)_Gtest_kde_$(file_title)_θ$(θ_in).png"
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
        plt = plot(x->G_u(x), x_s, x_e, label=L"G(\tau, \delta u)",
                    color="indigo", linewidth = 2.5)
        plot!(x->Gu_pred(i, x), x_s, x_e, marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"G_{\theta}(\tau, \delta u)", linestyle=:dash, linewidth = 2.5)
        title!(L"\textrm{Comparing } G(\tau,z) \textrm{ and } \hat{G}_{\theta}(\tau,z)", titlefont=20)
        xlabel!(L"\textrm{Velocity increment}", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"G_u \textrm{ distribution}", ytickfontsize=12, yguidefontsize=20)
    end

    gif(anim, file_out, fps = ceil(Int, n_itrs/sim_time))
    println("**************** Animation Complete ***************")
end



function compare_eos(p_h)
    gr(size=(600,600))
    Pnn_comp(ρ) = re(p_h)([ρ])[1]
    rho_data = 0.80:0.01:1.20
    P_gt = Pres.(rho_data, c_gt, g)
    P_nn = Pnn_comp.(rho_data)

    plt = plot(rho_data, P_gt, label="P_gt", color="blue", linewidth = 2.5)
    plot!(rho_data, P_nn, marker=:x, markersize=4,
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

function compare_H(p_h)
    gr(size=(600,600))
    Hnn_comp(r) = H_nn(r, p_h)
    r_data = 0.0:0.01:(2*h)
    H_gt = H.(r_data, h)

    plt = plot(r_data, H_gt, label=L"H_{gt}", color="blue", linewidth = 2.5)
	plot!(x->Hnn_comp(x), 0.0, 2*h, marker=:x, markersize=4,
		  color="forestgreen", markercolor = :black,
		  label=L"H_{nn(\theta)}(r)", linestyle=:dash, linewidth = 2.5)

	title!(L"\textrm{Learning Derivative } (\partial_r W(r))/r", titlefont=18)
    xlabel!(L"r", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"H(r, h)", ytickfontsize=12, yguidefontsize=20)

    display(plt)
	savefig(plt, "./figures/Hnn_$(sens_method)_$(loss_method)_$(method)_$(file_title)_height$(height).png")
end



function animate_learning_EoS(n_itr, rho_data, P_gt, P_nn, sim_time=10)
    file_out = "./anim/eosNN_$(method)_$(file_title)_height$(height).mp4"
    gr(size=(600,600))
    println("**************** Animating ***************")
    anim = @animate for i ∈ 1 : n_itr
        plot(rho_data, P_gt, label=L"P_{gt}", color="blue", linewidth = 2.5)
        plot!(rho_data, P_nn[i,:], marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"P_{nn(\theta)}(\rho)", linestyle=:dash, linewidth = 2.5)

        title!(L"\textrm{Learning EoS } P(\rho) ", titlefont=20)
        xlabel!(L" \rho", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"P(\rho)", ytickfontsize=12, yguidefontsize=20)
    end
    gif(anim, file_out, fps = ceil(Int, n_itr/sim_time))
    println("**************** Animation Complete ***************")
end


function animate_learning_W(n_itr, r_data, W_gt, W_nn_data, sim_time=10)
    file_out = "./anim/Wnn_$(method)_$(file_title)_height$(height).mp4"
    gr(size=(600,600))
    println("**************** Animating ***************")
    anim = @animate for i ∈ 1 : n_itr
        plot(r_data, W_gt, label=L"W_{gt}", color="blue", linewidth = 2.5)
        plot!(r_data, W_nn_data[i,:], marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"W_{nn}(\theta)(r)", linestyle=:dash, linewidth = 2.5)

        title!(L"\textrm{Learning Smoothing Kernel} W(r) ", titlefont=20)
        xlabel!(L" r", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"W(r, h)", ytickfontsize=12, yguidefontsize=20)
    end
    gif(anim, file_out, fps = ceil(Int, n_itr/sim_time))
    println("**************** Animation Complete ***************")
end

function animate_learning_H(n_itr, r_data, H_gt, H_nn_data, sim_time=10)
    file_out = "./anim/Hnn_$(method)_$(file_title)_height$(height).mp4"
    gr(size=(600,600))
    println("**************** Animating ***************")
    anim = @animate for i ∈ 1 : n_itr
        plot(r_data, H_gt, label=L"H_{gt}", color="blue", linewidth = 2.5)
        plot!(r_data, H_nn_data[i,:], marker=:x, markersize=4,
              color="forestgreen", markercolor = :black,
              label=L"H_{nn}(\theta)(r)", linestyle=:dash, linewidth = 2.5)

        title!(L"\textrm{Learning Derivative } (\partial_r W(r))/r ", titlefont=20)
        xlabel!(L" r", xtickfontsize=12, xguidefontsize=20)
        ylabel!(L"H(r, h)", ytickfontsize=12, yguidefontsize=20)
    end
    gif(anim, file_out, fps = ceil(Int, n_itr/sim_time))
    println("**************** Animation Complete ***************")
end


"""

=========================== Loss Plots =========================================

"""


function plot_loss_itr()
    println(" ===== Plotting Loss =====")
    gr(size=(500,500))
    xs = 1 : vis_rate : (vis_rate * size(L_out)[1])
    plt = plot(xs, L_out, label="$(loss_method)", color="blue", yaxis=:log, linewidth = 2.5)

    title!(L"\textrm{WC SPH AV:  N = } %$N \textrm{, time steps = } %$T", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{Loss}", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    savefig(plt, "./figures/loss_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end


function plot_Lg_itr()
    println(" ===== Plotting Generalization Loss =====")
    gr(size=(500,500))
    xs = 1 : vis_rate : (vis_rate * size(Lg_out)[1])
    plt = plot(xs, Lg_out, label="KL", color="blue", yaxis=:log, linewidth = 2.5)

    title!(L"\textrm{WC SPH AV: N = } %$N \textrm{, time steps = } %$T", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{Generalization - Error}", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    savefig(plt, "./figures/gen_loss_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end

function plot_Lg2_itr()
    println(" ===== Plotting Generalization Loss =====")
    gr(size=(500,500))
    xs = 1 : vis_rate : (vis_rate * size(Lg2_out)[1])
    plt = plot(xs, Lg2_out, label="$(loss_method)", color="blue", yaxis=:log, linewidth = 2.5)

    title!(L"\textrm{WC SPH AV: N = } %$N \textrm{, time steps = } %$T", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"\textrm{Generalization Error}", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    savefig(plt, "./figures/gen2_loss_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end



function plot_rot_itr()
    println(" ===== Plotting rotation error =====")
    gr(size=(500,500))
    xs = 1 : vis_rate : (vis_rate * size(rot_QF)[1])

    plt = plot(xs, rot_QF, label="||QF - F(QY)||", color="blue", yaxis=:log, linewidth = 2.25)
    title!(L"\textrm{WCSPH: Rotational - error}", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||QF - F(QY)||_2", ytickfontsize=10, yguidefontsize=16)

    plt2 = plot(xs, rot_RF, label="||RF - F(RY)||", color="blue", yaxis=:log, linewidth = 2.25)
    title!(L"\textrm{WCSPH: Rotational - error}", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||RF(Y) - F(RY)||_2", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    display(plt2)
    savefig(plt, "./figures/Qf_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
    savefig(plt2, "./figures/Rf_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end

function plot_T()
    println(" ===== Plotting Translation error =====")
    gr(size=(500,500))
    xs = 1 : vis_rate : (vis_rate * size(Tx)[1])
    plt = plot(xs, Tx, label="||QF - F(QY)||", color="blue", yaxis=:log, linewidth = 2.25)
    plt2 = plot(xs, Ty, label="||RF - F(RY)||", color="blue", yaxis=:log, linewidth = 2.25)

    title!(L"\textrm{WCSPH: Translational - error}", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||F(X-s) - F(X)||_2", ytickfontsize=10, yguidefontsize=16)

    display(plt)
    display(plt2)
    savefig(plt, "./figures/Tx_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
    savefig(plt2, "./figures/Ty_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end


function plot_galilean_itr()
    println(" ===== Plotting Translational error =====")
    gr(size=(500,500))
    xs = 1 : vis_rate : (vis_rate * size(galilean_inv)[1])

    plt2 = plot(xs, rot_RF, label="||F(x-s) - F(x)||", color="blue", yaxis=:log, linewidth = 2.25)
    title!(L"\textrm{WCSPH: Translational - error}", titlefont=16)
    xlabel!(L"\textrm{Iterations}", xtickfontsize=10, xguidefontsize=16)
    ylabel!(L"||F(Y - s) - F(Y)||_2", ytickfontsize=10, yguidefontsize=16)

    display(plt2)
    savefig(plt2, "./figures/Gal_error_$(sens_method)_$(loss_method)_$(method)_$(file_title).png")
end


# function compare_eos_W_H(rho_data, r_data, p_h)
# 	if method == "eos_nn"
# 		Pnn_comp(ρ) = re(p_h)([ρ])[1];
# 		P_nn[k, :] = Pnn_comp.(rho_data);
# 		compare_eos(p_h);
# 	end
# 	if method == "Wnn"
# 		Wnn_comp(r) = W_nn(r, p_h);
# 		W_nn_data[k, :] = Wnn_comp.(r_data);
# 		compare_W(p_h);
# 	end
# 	if method == "Hnn"
# 		Hnn_comp(r) = H_nn(r, p_h);
# 		H_nn_data[k, :] = Hnn_comp.(r_data);
# 		compare_H(p_h);
# 	end
# end
