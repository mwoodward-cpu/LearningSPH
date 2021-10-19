

"""
					Senstivity analysis NODE model

"""


n_list = 22;
n_features = 6*n_list;

NN = Chain(
      Dense(n_features, height, tanh), #R^n -> R^h
      Dense(height, height, tanh), #R^h -> R^h
      Dense(height, 3)        #R^h -> R^D
    )


p_hat, re = Flux.destructure(NN)   #flatten nn params
f_nn(x) = NN(x)  #input needs to be [float]
n_params = size(p_hat)[1];


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



function compute_Π(XX, VV, rhoi, rhoj, α, β, h, c)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    Π = (-α*c*μ + β * μ^2)/((rhoi + rhoj)/2)
  end
  if (XX'*VV >= 0)
    Π = 0.0
  end
  return Π
end

function Pres(rho, c, g)
  return c^2 * (rho^g - 1.) / g ;
end


# P(rho) / rho^2
function P_d_rho2(rho, Pres, c, g)
  return  Pres(rho, c, g) / (rho^2);
end



fxnn_update(x, pn) = re(pn)(x)[1]
fynn_update(x, pn) = re(pn)(x)[2]
fznn_update(x, pn) = re(pn)(x)[3]


#reverse mode (best for |pn| >> 100)
∂fx_∂θk(x, pn) = gradient(() -> fxnn_update(x, pn), params(pn))[params(pn)[1]]
∂fy_∂θk(x, pn) = gradient(() -> fynn_update(x, pn), params(pn))[params(pn)[1]]
∂fz_∂θk(x, pn) = gradient(() -> fznn_update(x, pn), params(pn))[params(pn)[1]]


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
  fxij_nn(x) = fxnn_update(x, p_in);
  fyij_nn(x) = fynn_update(x, p_in);
  fzij_nn(x) = fznn_update(x, p_in);
  fij_nn(x) = re(p_in)(x);

  #Forward mode: (best for |x| < 100)
  ∂fxij_∂x(x) = ForwardDiff.gradient(x ->fxij_nn(x), x)
  ∂fyij_∂x(x) = ForwardDiff.gradient(x ->fyij_nn(x), x)
  ∂fzij_∂x(x) = ForwardDiff.gradient(x ->fzij_nn(x), x)


  ∂F_p = zeros(N, 2*D, n_params);
  ∂F_x = zeros(N, 2*D, 2*D);
  A = zeros(N,D); rho = ones(N);
  tke = 0.5*mean(V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2);

  XX, VV = obtain_disv(X, V, N, D);
  XX, VV = sorted_disv(XX, VV);
  #Computing A and the sensitivities
  for n in 1 : N
    feature_ij = vcat(XX[n, :, 1], XX[n, :, 2], XX[n, :, 3], VV[n, :, 1], VV[n, :, 2], VV[n, :, 3]);
    accl = fij_nn(feature_ij)
    for i in 1 : D
      A[n, i] = accl[i] #+ θ * (V[n, i] - mean(V[:, i]))
    end

    ∂F_p[n, 4, :] .+= ∂fx_∂θk(feature_ij, p_in)
    ∂F_p[n, 5, :] .+= ∂fy_∂θk(feature_ij, p_in)
    ∂F_p[n, 6, :] .+= ∂fz_∂θk(feature_ij, p_in)

    ∂f_x = ∂fxij_∂x(feature_ij);
    ∂f_y = ∂fyij_∂x(feature_ij);
    ∂f_z = ∂fzij_∂x(feature_ij);

    #∂ₓF^x:
    ∂F_x[n, 4, 1] = sum(∂f_x[1:n_list])
    #∂yF^x:
    ∂F_x[n, 4, 2] = sum(∂f_x[n_list + 1 : 2*n_list])
    #∂zF^x:
    ∂F_x[n, 4, 3] = sum(∂f_x[2*n_list + 1 : 3*n_list]) #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]))
    #∂uF^x:
    ∂F_x[n, 4, 4] = sum(∂f_x[3*n_list + 1 : 4*n_list])
    #∂vF^x:
    ∂F_x[n, 4, 5] = sum(∂f_x[4*n_list + 1 : 5*n_list])
    #∂wF^x:
    ∂F_x[n, 4, 6] = sum(∂f_x[5*n_list + 1 : 6*n_list])

    #∂ₓF^y:
    ∂F_x[n, 5, 1] = sum(∂f_y[1:n_list])
    #∂yF^y:
    ∂F_x[n, 5, 2] = sum(∂f_y[n_list + 1 : 2*n_list])
    #∂zF^y:
    ∂F_x[n, 5, 3] = sum(∂f_y[2*n_list + 1 : 3*n_list]) #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]))
    #∂uF^y:
    ∂F_x[n, 5, 4] = sum(∂f_y[3*n_list + 1 : 4*n_list])
    #∂vF^y:
    ∂F_x[n, 5, 5] = sum(∂f_y[4*n_list + 1 : 5*n_list])
    #∂wF^y:
    ∂F_x[n, 5, 6] = sum(∂f_y[5*n_list + 1 : 6*n_list])

    #∂ₓF^z:
    ∂F_x[n, 6, 1] = sum(∂f_z[1:n_list])
    #∂yF^z:
    ∂F_x[n, 6, 2] = sum(∂f_z[n_list + 1 : 2*n_list])
    #∂zF^z:
    ∂F_x[n, 6, 3] = sum(∂f_z[2*n_list + 1 : 3*n_list]) #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]))
    #∂uF^z:
    ∂F_x[n, 6, 4] = sum(∂f_z[3*n_list + 1 : 4*n_list])
    #∂vF^z:
    ∂F_x[n, 6, 5] = sum(∂f_z[4*n_list + 1 : 5*n_list]);
    #∂wF^z:
    ∂F_x[n, 6, 6] = sum(∂f_z[5*n_list + 1 : 6*n_list]);

  end
  for i in 1 : D
    A[:, i] += (θ/tke) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho, ∂F_x, ∂F_p
end




#---------------------Obtain A half step function


function obtain_sph_AV_A(X, V, p_in)
  fij_nn(x) = re(p_in)(x)
  A = zeros(N, D); rho = ones(N);
  tke = 0.5*mean(V[: ,1].^2 .+ V[: ,2].^2 .+ V[: ,3].^2);

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

  #Computing A and the sensitivities
  for n in 1 : N
    feature_ij = vcat(XX[n, :, 1], XX[n, :, 2], XX[n, :, 3], VV[n, :, 1], VV[n, :, 2], VV[n, :, 3]);
    # println("size of ∂fxij_∂x = ", size(∂fxij_∂x(feature_ij)))
    accl = fij_nn(feature_ij)
    for i in 1 : D
      A[n, i] = accl[i] #+ θ * (V[n, i] - mean(V[:, i]))
    end
  end
  for i in 1 : D
    A[:, i] += (θ/tke) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho
end
