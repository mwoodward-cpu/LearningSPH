"""
Applying KDE for estimating pdf of single particle dispersion
    statistic and velocity increment over a sample size of nt_samp*N

    for now, we just use nt_samp = 1
    Goals:
        -- Use derivate of G in computing ∇L in probabilistic learning file
        -- Formulate KL divergence for continuous case

    NOTES:
        -- Currently testing, need to clean up after
"""


using LinearAlgebra, Statistics, LaTeXStrings
# using QuadGK, NPZ, Plots

num_tau = T                         #number of time steps for computing single particle statistics
nt_samp = 1;                        #number of random samples in time for obtaining converged Gt G
M = nt_samp * N                     #total number of particles used in forming G_gt with kde
N_samp = ceil(Int, 1/nt_samp*M)     #total number of samples of true distribution (must be less than M)



"""
############  Single Particle Dispersion Statistic  ###############
"""

function obtain_sampled_gt_disp_vel_inc(traj_gt, vels_gt, nt_samp)
    rand_samp = zeros(nt_samp)
    for i ∈ 1 : nt_samp
        # rand_samp[i] = floor(Int, 400*rand() + 1)
        rand_samp[i] = nt_samp
    end

    M = nt_samp * N

    rand_samp = floor.(Int, rand_samp)
    r0_gt = zeros(nt_samp, N, D)
    v0_gt = zeros(nt_samp, N, D)

    r_gt = zeros(num_tau, nt_samp, N, D)
    v_gt = zeros(num_tau, nt_samp, N, D)
    Diff = zeros(num_tau, M)
    Diff_vec = zeros(num_tau, M, D)
    Vel_inc = zeros(num_tau, M, D);
    ∂Diff_∂xi = zeros(num_tau, M, D)

    for i ∈ 1 : nt_samp
        for τ ∈ 1 : num_tau
            r_gt[τ, i, :, :] = traj_gt[rand_samp[i] + τ, :, :]
            v_gt[τ, i, :, :] = vels_gt[rand_samp[i] + τ, :, :]
        end
    end

    for i ∈ 1 : nt_samp
        r0_gt[i, :, :] = traj_gt[rand_samp[i], :, :]
        v0_gt[i, :, :] = vels_gt[rand_samp[i], :, :]
    end

    r_gt = reshape(r_gt, (num_tau, M, D))
    v_gt = reshape(v_gt, (num_tau, M, D))
    r0_gt =  reshape(r0_gt, (M, D))
    v0_gt =  reshape(v0_gt, (M, D))

    for m in 1 : M
        for τ in 1 : num_tau
            for i in 1 : D
                Diff_vec[τ, m, i] = r_gt[τ, m, i] - r0_gt[m, i] - dt * τ * v0_gt[m, i]
                Vel_inc[τ, m, i] = v_gt[τ, m, i] - v0_gt[m, i];
                while (Diff_vec[τ, m, i] > pi)   Diff_vec[τ, m, i] -= 2. * pi;   end
                while (Diff_vec[τ, m, i] < -pi)   Diff_vec[τ, m, i] += 2. * pi;   end
            end
            Diff[τ, m] = sqrt(sum((Diff_vec[τ, m, :]).^2))
            for i in 1 : D
                ∂Diff_∂xi[τ, m, i] = Diff_vec[τ, m, i] / Diff[τ, m]
            end
            # Diff_vec = reshape(Diff_vec, (num_tau,2*M))
        end
    end
    return Diff, Vel_inc, ∂Diff_∂xi
end


function obtain_gt_dists(traj_gt, vels_gt)
    Diff_gt_vec = zeros(num_tau, N, D)
    Diff_gt = zeros(num_tau, N)
    Vel_inc_gt = zeros(num_tau, N, D)
    r0 = traj_gt[1,:,:]; v0 = vels_gt[1,:,:]
    for m in 1 : N
        for t in 1 : (num_tau)
            for i in 1 : D
                Diff_gt_vec[t, m, i] = traj_gt[1 + t, m, i] - r0[m, i] - dt * t * v0[m, i]
                Vel_inc_gt[t, m, i] = vels_gt[1 + t, m, i] - v0[m, i];
                while (Diff_gt_vec[t, m, i] > pi)   Diff_gt_vec[t, m, i] -= 2. * pi;   end
                while (Diff_gt_vec[t, m, i] < -pi)   Diff_gt_vec[t, m, i] += 2. * pi;   end
            end
            Diff_gt[t, m] = sqrt(sum((Diff_gt_vec[t, m, :]).^2))
        end
    end
    return Diff_gt, Vel_inc_gt
end


function obtain_pred_dists(traj_pred, vels_pred, r0, v0)
    Diff_pred_vec = zeros(num_tau, N, D)
    Diff_pred = zeros(num_tau, N)
    Vel_inc_pred = zeros(num_tau, N, D)
    for m in 1 : N
        for t in 1 : (num_tau)
            for i in 1 : D
                Diff_pred_vec[t, m, i] = traj_pred[1 + t, m, i] - r0[m, i] - dt * t * v0[m, i]
                Vel_inc_pred[t, m, i] = vels_pred[1 + t, m, i] - v0[m, i];
                while (Diff_pred_vec[t, m, i] > pi)   Diff_pred_vec[t, m, i] -= 2. * pi;   end
                while (Diff_pred_vec[t, m, i] < -pi)   Diff_pred_vec[t, m, i] += 2. * pi;   end
            end
            Diff_pred[t, m] = sqrt(sum((Diff_pred_vec[t, m, :]).^2))
        end
    end
    return Diff_pred, Vel_inc_pred
end


function obtain_pred_dists_step(traj_pred, vels_pred, r0, v0)
    Diff_pred_vec = zeros(N, D)
    Diff_pred = zeros(N)
    Vel_inc_pred = zeros(N, D)
    for m in 1 : N
            for i in 1 : D
                Diff_pred_vec[m, i] = traj_pred[m, i] - r0[m, i] - dt * v0[m, i]
                Vel_inc_pred[m, i] = vels_pred[m, i] - v0[m, i];
                while (Diff_pred_vec[m, i] > pi)   Diff_pred_vec[m, i] -= 2. * pi;   end
                while (Diff_pred_vec[m, i] < -pi)   Diff_pred_vec[m, i] += 2. * pi;   end
            end
        Diff_pred[m] = sqrt(sum((Diff_pred_vec[m, :]).^2))
    end
    return Diff_pred, Vel_inc_pred
end


"""
============== KDE ===================
"""


#---KDE smoothing kernel for density estimation (both for gt and pred data)
K(x) = 1/(sqrt(2*pi))*exp(-x^2/2) #guassian kernel (best results so far)
# K(x) = maximum([1 - abs(x), 0])     #triangle (produces frequency polygon)
# K(x) = 3/4*maximum([1 - x^2, 0])   #Epanechnikov
K_prime(x) = ForwardDiff.derivative(x -> K(x), x) #(used in computing ∇L)



function kde(z, z_data)
    Nm = length(z_data);
        #silvermans rule (h_kde = 0.9)
    h = h_kde * std(z_data)*Nm^(-1/5)
    return 1/(Nm*h) * sum(K.((z_data .- z)/h))
end

function obtain_h_kde(z_data)
    Nm = length(z_data);
    h = h_kde * std(z_data)*Nm^(-1/5)
    return h
end


#---------Obtain ground truth G:

# Diff_gt, Vel_inc_gt, ∂Diff_∂x = obtain_gt_disp_vel_inc(traj_gt, vels_gt, nt_samp)
Diff_gt, Vel_inc_gt = obtain_gt_dists(traj_gt, vels_gt)
G(x) = kde(x, Diff_gt[T, :]);         hd_gt = obtain_h_kde(Diff_gt[T, :])
G_u(x) = kde(x, Vel_inc_gt[T, :, 1]); hu_kde_gt = obtain_h_kde(Vel_inc_gt[T, :, 1])
G_v(x) = kde(x, Vel_inc_gt[T, :, 2]); hv_kde_gt = obtain_h_kde(Vel_inc_gt[T, :, 2])

Gt_u(x, t) = kde(x, Vel_inc_gt[t, :, 1]); hut_kde_gt(t) = obtain_h_kde(Vel_inc_gt[t, :, 1]);
Gt_v(x, t) = kde(x, Vel_inc_gt[t, :, 2]); hvt_kde_gt(t) = obtain_h_kde(Vel_inc_gt[t, :, 2]);

# G_w(x) = kde(x, Vel_inc_gt[T, :, 3]); hw_kde_gt = obtain_h_kde(Vel_inc_gt[T, :, 3])


function plotting_G_kde(G)
    gr(size=(500,400))
    x_s = -0.0002
    x_e = 0.002
    x_e1 = 0.002
    # x_g = x_s:0.0001:x_e
    plt = plot(x->G(x), x_s, x_e1, label="G(τ, d)", color="maroon4", linewidth = 2.5)
    display(plt)

    title!(L"\textrm{KDE of } G_d(\tau, z) \textrm{ with N = } %$N \textrm{ samples}", titlefont=17)
    xlabel!(L"|| \mathbf{r}_i(\tau) - \mathbf{r}_i(0) - \tau \mathbf{v}_i(0)||_2")
    ylabel!(L"\hat{G}", ytickfontsize=12, yguidefontsize=20)

    display(plt)
    savefig(plt, "./figures/G_kde_D$(D)_N$(N)_M$(M).png")
end

# plotting_G_kde(G)


function plotting_Gvel_kde(width=0.1)
    gr(size=(500,500))
    x_s = -width
    x_e = width
    plt = plot(x->G_u(x), x_s, x_e, label=L"G_v(\tau, d)", color="indigo", linewidth = 2.5)
    display(plt)

    title!(L"\textrm{KDE of } G_v(\tau, z) \textrm{ with N = } %$N \textrm{ samples}", titlefont=17)
    xlabel!(L"\textrm{velocity increment } u_i(\tau) - u_i(0)", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"G_v \textrm{ distribution}", ytickfontsize=12, yguidefontsize=20)

    display(plt)
    savefig(plt, "./figures/Gv_kde_D$(D)_N$(N).png")
end

# plotting_Gvel_kde(window)



"""
============== KL Divergence ===================
        Testing continuous form
          with integration


          Move to main code and remove once finished
"""



# r0 = traj_gt[1, :, :]
# v0 = vels_gt[1, :, :]
# Diff_pred = obtain_pred_dist(traj_gt[1:11, :, :], r0, v0)
#
# function kl_fixed_τ(Diff_gt, Diff_pred, τ)
#     #Forward KL: data is sampled from GT distritubtion
#     L = 0.0
#     xe = maximum(Diff_gt[τ, :])
#     G_pred(τ, d) = kde(d, Diff_pred[τ, :])
#     f(x) = G(x) * log(G(x) / G_pred(τ, x))
#     L, err = quadgk(f, 0, xe, rtol=1e-4)
#     return L
# end
#
#
# function kl_over_τ(Diff_gt, Diff_pred, T)
#     L = 0.0
#     for τ in 1 : T
#         L += kl_fixed_τ(Diff_gt, Diff_pred, τ)
#     end
#     return L
# end







 #see main code (later we plan to test loss functions)

# function kldivergence1(Diff_gt, Diff_pred, T)
#     #Forward KL: data is sampled from GT distritubtion
#     L = 0.0
#     M = size(Diff_gt)[2]
#     G_pred(τ, d) = kde(d, Diff_pred[τ, :])
#     for τ in 1 : T
#         for i in 1 : M
#             Ggt_i = G(τ, Diff_gt[τ, i])
#             Gθ_i = G_pred(τ, Diff_gt[τ, i])
#             L += 1/(M*T) * Ggt_i * log(Ggt_i / Gθ_i)
#         end
#     end
#     return L
# end
#
#
# function ∂kl_∂d(Diff_pred, τ, d)
#     #Obtain derivative of kl wrt input d using AD: forward mode
#     G_pred(τ, d) = kde(d, Diff_pred[τ, :])
#     kli(τ, d) = G(τ, d) * log(G(τ, d) / G_pred(τ, d))
#     return ForwardDiff.derivative(d -> kli(τ, d), d)
# end



"""
============== Cross Entropy loss ===================
"""
#
# function l1_i(τ, d)
#     return -log(G(τ, d))
# end
#
# function l2_i(τ, d)
#     return -log(G(τ, d))
# end
#
# function cross_entropy1(data, T)
#     #Data is from prediction step
#     L = 0.0
#     for τ in 1 : T
#         for i in 1 : size(data)[2]
#             L += l1_i(τ, data[τ, i])
#         end
#     end
#     return L
# end
#
#
# function cross_entropy2(data, T)
#     #Data is from true distribution
#     L = 0.0
#     for τ in 1 : T
#         for i in 1 : size(data)[2]
#             L += l2_i(τ, data[τ, i])
#         end
#     end
#     return L
# end
#

"""
============== Maximum loglikelihood estimation ===================
"""
#
# function mle_i(τ, d)
#     return -log(G_pred_test(τ, d))
# end
#
#
# function MLE(data, T)
#     #data is sampled from true distribution
#     L = 0.0
#     M = size(data)[2]
#     for τ in 1 : T
#         for i in 1 : M
#             L += 1/(T*M) * mle_i(τ, data[τ, i])
#         end
#     end
#     return L
# end
