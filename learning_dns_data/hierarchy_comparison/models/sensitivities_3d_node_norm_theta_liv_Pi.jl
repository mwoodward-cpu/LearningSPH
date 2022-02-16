

"""
					Senstivity analysis NODE model

"""


n_list = 20;
n_features = 6*n_list;
n_params = size(p_fin)[1];


#------- Smoothing kernel and necessary derivatives
# sigma = (10. / (7. * pi * h * h));
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


#reverse mode (best for |pn| >> 100)
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

n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;
function obtain_disv(X, V, N, D)
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

  VV = zeros(N, n_list, D);
  XX = zeros(N, n_list, D);
  for n in 1 : N
    idx_neigh = 0
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
            idx_neigh += 1
            if (idx_neigh > n_list)   close = false; break;   end
            close = true; r2 = 0.;
            for i in 1 : D
              XX[n, idx_neigh, i] = X[n, i] - X[n2, i];
              VV[n, idx_neigh, i] = V[n, i] - V[n2, i];
              while (XX[n, idx_neigh, i] > pi)   XX[n, idx_neigh, i] -= 2. * pi;   end
              while (XX[n, idx_neigh, i] < -pi)   XX[n, idx_neigh, i] += 2. * pi;   end
              r2 += XX[n, idx_neigh, i] * XX[n, idx_neigh, i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
          end
        end
      end
    end
  end
  return XX, VV
end

function sorted_disv(XX, VV)
  XX_ = sort(abs.(XX), dims=2, rev=true)[:, 1:n_list, :];
  VV_ = sort(abs.(VV), dims=2, rev=true)[:, 1:n_list, :];
  return XX_, VV_
end

function obtain_sph_diff_A_model_deriv(X, V, p_in)
  pnn_in = p_in[1:(end-1)]; θ_in = p_in[end];
  fxij_nn(x) = fxnn_update(x, pnn_in);
  fyij_nn(x) = fynn_update(x, pnn_in);
  fzij_nn(x) = fznn_update(x, pnn_in);
  fij_nn(x) = re(pnn_in)(x);

  #Forward mode: (best for |x| < 100)
  ∂fxij_∂x(x) = ForwardDiff.gradient(x ->fxij_nn(x), x)
  ∂fyij_∂x(x) = ForwardDiff.gradient(x ->fyij_nn(x), x)
  ∂fzij_∂x(x) = ForwardDiff.gradient(x ->fzij_nn(x), x)


  ∂F_p = zeros(N, 2*D, n_params);
  ∂F_x = zeros(N, 2*D, 2*D);
  A = zeros(N,D); rho = ones(N);
  tke = 0.5*mean(rho .* (V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2));

  XX, VV = obtain_disv(X, V, N, D);
  XX, VV = sorted_disv(XX, VV);
  #Computing A and the sensitivities
  for n in 1 : N
    feature_xx = vcat(XX[n, :, 1], XX[n, :, 2], XX[n, :, 3]);
    feature_xxnorm = (feature_xx .- minimum(feature_xx))/(maximum(feature_xx) - minimum(feature_xx));
    feature_vv = vcat(VV[n, :, 1], VV[n, :, 2], VV[n, :, 3])
    feature_vvnorm = (feature_vv .- minimum(feature_vv))/(maximum(feature_vv) - minimum(feature_vv));

    feature_ij = vcat(feature_xxnorm, feature_vvnorm);
    accl = fij_nn(feature_ij)
    for i in 1 : D
      A[n, i] = accl[i] #+ θ * (V[n, i] - mean(V[:, i]))
    end

    ∂F_p[n, 4, 1:(end-1)] .= ∂fx_∂θk(feature_ij, pnn_in)
    ∂F_p[n, 4, end] = ∂fij_p(V[n,:], tke, θ_in, rho[n], 1)
    ∂F_p[n, 5, 1:(end-1)] .= ∂fy_∂θk(feature_ij, pnn_in)
    ∂F_p[n, 5, end] = ∂fij_p(V[n,:], tke, θ_in, rho[n], 2)
    ∂F_p[n, 6, 1:(end-1)] .= ∂fz_∂θk(feature_ij, pnn_in)
    ∂F_p[n, 6, end] = ∂fij_p(V[n,:], tke, θ_in, rho[n], 3)

    ∂f_x = ∂fxij_∂x(feature_ij);
    ∂f_y = ∂fyij_∂x(feature_ij);
    ∂f_z = ∂fzij_∂x(feature_ij);

    #∂ₓF^x:
    ∂F_x[n, 4, 1] = sum(∂f_x[1:n_list])
    #∂yF^x:
    ∂F_x[n, 4, 2] = sum(∂f_x[n_list + 1 : 2*n_list])
    #∂zF^x:
    ∂F_x[n, 4, 3] = sum(∂f_x[2*n_list + 1 : 3*n_list])
    #∂uF^x:
    ∂F_x[n, 4, 4] = sum(∂f_x[3*n_list + 1 : 4*n_list]) #+ θ_in / (2*tke)
    #∂vF^x:
    ∂F_x[n, 4, 5] = sum(∂f_x[4*n_list + 1 : 5*n_list])
    #∂wF^x:
    ∂F_x[n, 4, 6] = sum(∂f_x[5*n_list + 1 : 6*n_list])

    #∂ₓF^y:
    ∂F_x[n, 5, 1] = sum(∂f_y[1:n_list])
    #∂yF^y:
    ∂F_x[n, 5, 2] = sum(∂f_y[n_list + 1 : 2*n_list])
    #∂zF^y:
    ∂F_x[n, 5, 3] = sum(∂f_y[2*n_list + 1 : 3*n_list])
    #∂uF^y:
    ∂F_x[n, 5, 4] = sum(∂f_y[3*n_list + 1 : 4*n_list])
    #∂vF^y:
    ∂F_x[n, 5, 5] = sum(∂f_y[4*n_list + 1 : 5*n_list]) #+ θ_in / (2*tke)
    #∂wF^y:
    ∂F_x[n, 5, 6] = sum(∂f_y[5*n_list + 1 : 6*n_list])

    #∂ₓF^z:
    ∂F_x[n, 6, 1] = sum(∂f_z[1:n_list])
    #∂yF^z:
    ∂F_x[n, 6, 2] = sum(∂f_z[n_list + 1 : 2*n_list])
    #∂zF^z:
    ∂F_x[n, 6, 3] = sum(∂f_z[2*n_list + 1 : 3*n_list])
    #∂uF^z:
    ∂F_x[n, 6, 4] = sum(∂f_z[3*n_list + 1 : 4*n_list])
    #∂vF^z:
    ∂F_x[n, 6, 5] = sum(∂f_z[4*n_list + 1 : 5*n_list]);
    #∂wF^z:
    ∂F_x[n, 6, 6] = sum(∂f_z[5*n_list + 1 : 6*n_list]) #+ θ_in / (2*tke)

  end
  if extern_f=="determistic"
    for i in 1 : D
      A[:, i] += (θ_in/(2*tke)) * rho .* (V[:, i])
    end
  end
  return A, rho, ∂F_x, ∂F_p
end




#---------------------Obtain A half step function


function obtain_sph_AV_A(X, V, p_in)
  pnn_in = p_in[1:(end-1)]; θ_in = p_in[end];
  fij_nn(x) = re(pnn_in)(x)
  A = zeros(N, D); rho = zeros(N);
  # tke = 0.5*mean(V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2);

  XX, VV = obtain_disv(X, V, N, D);
  XX, VV = sorted_disv(XX, VV);


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

    XX_v = zeros(D);
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
                XX_v[i] = X[n, i] - X[n2, i];
                while (XX_v[i] > pi)   XX_v[i] -= 2. * pi;   end
                while (XX_v[i] < -pi)   XX_v[i] += 2. * pi;   end
                r2 += XX_v[i] * XX_v[i];
                if (r2 > 4. * h * h)   close = false; break;   end
              end
              if (close)
                tmp = m * W(sqrt(r2), h); rho[n] += tmp;
              end
            end
          end
      end   end
    end

  tke = 0.5*mean(rho .* (V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2));
  #Computing A and the sensitivities
  for n in 1 : N
    feature_xx = vcat(XX[n, :, 1], XX[n, :, 2], XX[n, :, 3]);
    feature_xxnorm = (feature_xx .- minimum(feature_xx))/(maximum(feature_xx) - minimum(feature_xx));
    feature_vv = vcat(VV[n, :, 1], VV[n, :, 2], VV[n, :, 3])
    feature_vvnorm = (feature_vv .- minimum(feature_vv))/(maximum(feature_vv) - minimum(feature_vv));

    feature_ij = vcat(feature_xxnorm, feature_vvnorm);
    # feature_ij = vcat(feature_xx, feature_vv);

    accl = fij_nn(feature_ij);
    for i in 1 : D
      A[n, i] = accl[i] #+ θ * (V[n, i] - mean(V[:, i]))
    end
  end
  if extern_f=="determistic"
    for i in 1 : D
      A[:, i] += (θ_in/(2*tke)) * rho .* (V[:, i])
    end
  end
  return A, rho
end
