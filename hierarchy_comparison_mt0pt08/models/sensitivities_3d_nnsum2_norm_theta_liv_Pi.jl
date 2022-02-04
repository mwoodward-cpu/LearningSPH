

"""
				 Senstivity analysis NN Summand model

"""


n_features = 6;

# NN = Chain(
#       Dense(n_features, height, tanh), #R^1 -> R^h
#       Dense(height, height, tanh),
#       Dense(height, 3)        #R^h -> R^1
#     )
#
#
# p_hat, re = Flux.destructure(NN)   #flatten nn params
# f_nn(x) = NN(x)  #input needs to be [float]
# p_hat = vcat(p_hat, [θ_gt])
n_params = size(p_fin)[1]


#------- Smoothing kernel and necessary derivatives
# const sigma = (10. / (7. * pi * h * h));
sigma = 1/(pi*h^3)  #3D normalizing factor

function W(r, h)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

# H(r) = (d W / d r) / r
function H(r, h)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * (-3. + 9. * q / 4.) / h^2);
end


fxnn_update(x, pn) = re(pn)(x)[1]
fynn_update(x, pn) = re(pn)(x)[2]
fznn_update(x, pn) = re(pn)(x)[3]

#reverse mode (best for |pn| > 100)
∂fx_∂θk(x, pn) = gradient(() -> fxnn_update(x, pn), params(pn))[params(pn)[1]]
∂fy_∂θk(x, pn) = gradient(() -> fynn_update(x, pn), params(pn))[params(pn)[1]]
∂fz_∂θk(x, pn) = gradient(() -> fznn_update(x, pn), params(pn))[params(pn)[1]]

function fij_ext(Vn, tke, θ, ρn, i)
  f_ext = (θ / (2*tke)) * ρn * (Vn[i])
  return f_ext
end

function ∂fij_p(Vn, tke, θ, ρn, i)
    return ForwardDiff.derivative(θ -> fij_ext(Vn, tke, θ, ρn, i), θ)
end


#--------------Model derivatives wrt variables and params

n_hash = floor(Int, 2. * pi / h);
l_hash = 2. * pi / n_hash;

function obtain_sph_diff_A_model_deriv(X, V, p_in)
  pnn_in = p_in[1:(end-1)]; θ_in = p_in[end];
  fxij_nn(x) = fxnn_update(x, pnn_in);
  fyij_nn(x) = fynn_update(x, pnn_in);
  fzij_nn(x) = fznn_update(x, pnn_in);

  fij_nn(x) = re(pnn_in)(x)

  #Reverse mode: (best for |x| > 100)
  # ∂fxij_∂x(x) = gradient(fxij_nn, x)[1]
  # ∂fyij_∂x(x) = gradient(fyij_nn, x)[1]

  #Forward mode: (best for |x| < 100)
  ∂fxij_∂x(x) = ForwardDiff.gradient(x ->fxij_nn(x), x)
  ∂fyij_∂x(x) = ForwardDiff.gradient(x ->fyij_nn(x), x)
  ∂fzij_∂x(x) = ForwardDiff.gradient(x ->fzij_nn(x), x)

  rho = ones(N);
  ∂F_p = zeros(N, 2*D, n_params);
  ∂F_x = zeros(N, 2*D, 2*D);
  A = zeros(N,D);
  drhoi_dxi = zeros(N, D); drhoj_dxi = zeros(N, D);
  P = zeros(N); mPdrho2 = zeros(N);
  tke = 0.5*mean(V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2);

  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash, k in 1 : n_hash];
  for n in 1 : N
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1,
               floor(Int, X[n, 3] / l_hash) + 1], n);
  end

  VV = zeros(D);
  XX = zeros(D);
  for n in 1 : N
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1,
              floor(Int, X[n, 3] / l_hash) + 1];
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
              XX[i] = X[n, i] - X[n2, i];
              VV[i] = V[n, i] - V[n2, i];
              while (XX[i] > pi)   XX[i] -= 2. * pi;   end
              while (XX[i] < -pi)   XX[i] += 2. * pi;   end
              r2 += XX[i] * XX[i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
            if (close)
              tmp = m * W(sqrt(r2), h); rho[n] += tmp;
              feature_x = [XX[1], XX[2], XX[3]];
              # feature_xnorm = (feature_x .- minimum(feature_x))/(maximum(feature_x) - minimum(feature_x));
              feature_v = [VV[1], VV[2], VV[3]];
              #feature_vnorm = (feature_v .- minimum(feature_v))/(maximum(feature_v) - minimum(feature_v));
              feature_ij = vcat(feature_x, feature_v);
              accl = fij_nn(feature_ij)
              for i in 1 : D
                A[n, i] += accl[i]
              end

              """
              Compute model derivatives wrt parameters: ∂F/∂θ
              ∂F_p [N, D^2, n_params] (n, 3, :) x-component of F
                n_param = 1,2,3,4   ->   c_in, h, α, β = p_in
              """

              ∂F_p[n, 4, 1:(end-1)] .+= ∂fx_∂θk(feature_ij, pnn_in)
              ∂F_p[n, 5, 1:(end-1)] .+= ∂fy_∂θk(feature_ij, pnn_in)
              ∂F_p[n, 6, 1:(end-1)] .+= ∂fz_∂θk(feature_ij, pnn_in)


              """
              Compute jacobian: ∂F/∂x
              """

              ∂f_x = ∂fxij_∂x(feature_ij);
              ∂f_y = ∂fyij_∂x(feature_ij);
              ∂f_z = ∂fzij_∂x(feature_ij);


              #∂ₓF^x:
              ∂F_x[n, 4, 1] += ∂f_x[1];
              #∂yF^x:
              ∂F_x[n, 4, 2] += ∂f_x[2];
              #∂zF^x:
              ∂F_x[n, 4, 3] += ∂f_x[3];
              #∂uF^x:
              ∂F_x[n, 4, 4] += ∂f_x[4]
              #∂vF^x:
              ∂F_x[n, 4, 5] += ∂f_x[5];
              #∂wF^x:
              ∂F_x[n, 4, 6] += ∂f_x[6];

              #∂ₓF^y:
              ∂F_x[n, 5, 1] += ∂f_y[1];
              #∂yF^y:
              ∂F_x[n, 5, 2] += ∂f_y[2];
              #∂zF^y:
              ∂F_x[n, 5, 3] += ∂f_y[3];
              #∂uF^y:
              ∂F_x[n, 5, 4] += ∂f_y[4];
              #∂vF^y:
              ∂F_x[n, 5, 5] += ∂f_y[5];
              #∂wF^y:
              ∂F_x[n, 5, 6] += ∂f_y[6];

              #∂ₓF^z:
              ∂F_x[n, 6, 1] += ∂f_z[1];
              #∂yF^z:
              ∂F_x[n, 6, 2] += ∂f_z[2];
              #∂zF^z:
              ∂F_x[n, 6, 3] += ∂f_z[3];
              #∂uF^z:
              ∂F_x[n, 6, 4] += ∂f_z[4];
              #∂vF^z:
              ∂F_x[n, 6, 5] += ∂f_z[5];
              #∂wF^z:
              ∂F_x[n, 6, 6] += ∂f_z[6];
            end
          end
        end
    end   end
  end
  for n in 1 : N
    tke = 0.5*mean(rho .* (V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2));
    ∂F_p[n, 4, end] = ∂fij_p(V[n,:], tke, θ_in, rho[n], 1)
    ∂F_p[n, 5, end] = ∂fij_p(V[n,:], tke, θ_in, rho[n], 2)
    ∂F_p[n, 6, end] = ∂fij_p(V[n,:], tke, θ_in, rho[n], 3)
  end
  if extern_f=="determistic"
    for i in 1 : D
      A[:, i] += (θ_in/(2*tke)) * rho .* (V[:, i])
    end
  end
  return A, rho, ∂F_x, ∂F_p
end






#---------------------Obtain A half step function


# n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;
function obtain_sph_AV_A(X, V, p_in)
  pnn_in = p_in[1:(end-1)]; θ_in = p_in[end];
  fij_nn(x) = re(pnn_in)(x)
  A = zeros(N, D);
  rho = zeros(N);
  mPdrho2 = zeros(N);
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash, k in 1 : n_hash];
  for n in 1 : N
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1,
               floor(Int, X[n, 3] / l_hash) + 1], n);
  end

  XX = zeros(D); VV = zeros(D);
  for n in 1 : N
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1,
              floor(Int, X[n, 3] / l_hash) + 1];
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
              XX[i] = X[n, i] - X[n2, i];
              while (XX[i] > pi)   XX[i] -= 2. * pi;   end
              while (XX[i] < -pi)   XX[i] += 2. * pi;   end
              r2 += XX[i] * XX[i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
            if (close)
              tmp = m * W(sqrt(r2), h); rho[n] += tmp;
              feature_x = [XX[1], XX[2], XX[3]];
              # feature_xnorm = (feature_x .- minimum(feature_x))/(maximum(feature_x) - minimum(feature_x));
              feature_v = [VV[1], VV[2], VV[3]];
              # feature_vnorm = (feature_v .- minimum(feature_v))/(maximum(feature_v) - minimum(feature_v));
              feature_ij = vcat(feature_x, feature_v);
              accl = fij_nn(feature_ij)
              for i in 1 : D
                A[n, i] += accl[i]
              end
            end
          end
        end
    end   end
  end
  tke = 0.5*mean(rho .* (V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2));
   if extern_f=="determistic"
     for i in 1 : D
       A[:, i] += (θ_in/(2*tke)) * rho .* (V[:, i])
     end
   end
  return A, rho
end
