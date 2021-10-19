

"""
grad_p sens

"""


n_features = 3;

# NN = Chain(
#       Dense(n_features, height, tanh), #R^1 -> R^h
#       Dense(height, height, tanh),
#       Dense(height, 1)        #R^h -> R^1
#     )
#
#
# p_hat, re = Flux.destructure(NN)   #flatten nn params
# n_params = size(p_hat)[1]


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

function dH_dr_r(r, h)
  q = r/h
  if (q >= 2.0)
    return 0.0
  end
  if (q >= 1.0)
    #return 3.0 * U0_in * (2.0 - q)^2/(4.0*r^3*h) + 3 * U0_in * (2 - q)/(2*h^2*r^2)
    return sigma * 1/r * (3/(h*r^2) - 3/(4*h^3))
  end
  return sigma * (9)/(4*h^3 * r)
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
  return  Pres(rho, c, g) / rho^2;
end

#check
function ∂P_rhoi2_∂x(rhoi, ∂ρi_∂xi, c, g)
  Pi = Pres(rhoi, c, g)
  ∂Pi_∂x = c^2 * rhoi^(g-1) * ∂ρi_∂xi
  ∂P_ρ2i_∂x = (∂Pi_∂x * rhoi^2 - 2*Pi*rhoi*∂ρi_∂xi)/(rhoi^4)
  return ∂P_ρ2i_∂x
end

#check
function ∂P_rhoi2_∂y(rhoi, ∂ρi_∂yi, c, g)
  Pi = Pres(rhoi, c, g)
  ∂Pi_∂y = c^2 * rhoi^(g-1) * ∂ρi_∂yi
  ∂P_ρ2i_∂y = (∂Pi_∂y * rhoi^2 - 2*Pi*rhoi*∂ρi_∂yi)/(rhoi^4)
  return ∂P_ρ2i_∂y
end



#---------------∂Π/∂x

function ∂Π_x(XX, VV, c, h, α, β, rhoi, rhoj, ∂ρi_xi, ∂ρj_xi)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    ρij = 0.5*(rhoi + rhoj)
    gij = (sum(XX.^2) + 0.01*h^2)
    fij = (XX'*VV)
    ∂μ_∂x = h*(VV[1]*gij - 2*XX[1]*fij)/(gij^2)
    ∂ρij_x = 0.5* (∂ρi_xi + ∂ρj_xi)
    ∂Π_∂xi = ((-α*c*∂μ_∂x + 2*β*μ*∂μ_∂x)*ρij + (α*c*μ - β*μ^2)*∂ρij_x)/(ρij^2)
  end
  if (XX'*VV >= 0)
    ∂Π_∂xi = 0.0
  end
  return ∂Π_∂xi
end


function ∂Π_y(XX, VV, c, h, α, β, rhoi, rhoj, ∂ρi_yi, ∂ρj_yi)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    ρij = 0.5*(rhoi + rhoj)
    gij = (sum(XX.^2) + 0.01*h^2)
    fij = (XX'*VV)
    ∂μ_∂y = h*(VV[2]*gij - 2*XX[2]*fij)/(gij^2)
    ∂ρij_y = 0.5* (∂ρi_yi + ∂ρj_yi)
    ∂Π_∂yi = ((-α*c*∂μ_∂y + 2*β*μ*∂μ_∂y)*ρij + (α*c*μ - β*μ^2)*∂ρij_y)/(ρij^2)
  end
  if (XX'*VV >= 0)
    ∂Π_∂yi = 0.0
  end
  return ∂Π_∂yi
end

function ∂Π_z(XX, VV, c, h, α, β, rhoi, rhoj, ∂ρi_zi, ∂ρj_zi)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    ρij = 0.5*(rhoi + rhoj)
    gij = (sum(XX.^2) + 0.01*h^2)
    fij = (XX'*VV)
    ∂μ_∂z = h*(VV[3]*gij - 2*XX[3]*fij)/(gij^2)
    ∂ρij_z = 0.5* (∂ρi_zi + ∂ρj_zi)
    ∂Π_∂zi = ((-α*c*∂μ_∂z + 2*β*μ*∂μ_∂z)*ρij + (α*c*μ - β*μ^2)*∂ρij_z)/(ρij^2)
  end
  if (XX'*VV >= 0)
    ∂Π_∂zi = 0.0
  end
  return ∂Π_∂zi
end


function ∂Π_u(XX, VV, c, h, α, β, rhoi, rhoj)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    ρij = 0.5*(rhoi + rhoj)
    gij = (sum(XX.^2) + 0.01*h^2)
    ∂μ_∂u = h*(XX[1])/(gij)
    ∂Π_∂ui = ((-α*c*∂μ_∂u + 2*β*μ*∂μ_∂u))/(ρij)
  end
  if (XX'*VV >= 0)
    ∂Π_∂ui = 0.0
  end
  return ∂Π_∂ui
end

function ∂Π_v(XX, VV, c, h, α, β, rhoi, rhoj)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    ρij = 0.5*(rhoi + rhoj)
    gij = (sum(XX.^2) + 0.01*h^2)
    ∂μ_∂v = h*(XX[2])/(gij)
    ∂Π_∂vi = ((-α*c*∂μ_∂v + 2*β*μ*∂μ_∂v))/(ρij)
  end
  if (XX'*VV >= 0)
    ∂Π_∂vi = 0.0
  end
  return ∂Π_∂vi
end

function ∂Π_w(XX, VV, c, h, α, β, rhoi, rhoj)
  if (XX'*VV < 0)
    μ = h*(XX'*VV)/(sum(XX.^2) + 0.01*h^2)
    ρij = 0.5*(rhoi + rhoj)
    gij = (sum(XX.^2) + 0.01*h^2)
    ∂μ_∂w = h*(XX[3])/(gij)
    ∂Π_∂wi = ((-α*c*∂μ_∂w + 2*β*μ*∂μ_∂w))/(ρij)
  end
  if (XX'*VV >= 0)
    ∂Π_∂wi = 0.0
  end
  return ∂Π_∂wi
end

fnn_update(x, pn) = re(pn)(x)[1]
#reverse mode (best for |pn| > 100)
∂f_∂θk(x, pn) = gradient(() -> fnn_update(x, pn), params(pn))[params(pn)[1]]


#--------------Model derivatives wrt variables and params

n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;
function obtain_sph_diff_A_model_deriv(X, V, p_in)
  fij_nn(x) = fnn_update(x, p_in);
  #Forward mode: (best for |x| < 100)
  ∂fij_∂x(x) = ForwardDiff.gradient(x ->fij_nn(x)[1], x)

  rho = zeros(N);
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

  XX = zeros(D);
  # computing rho
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
          end
        end
    end   end   end
  end

  VV = zeros(D); XX = zeros(D);
  for n in 1 : N
    # A[n, :] = obtain_forcing_A(X[n, :]);
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

            close = true;   r2 = 0.;
            for i in 1 : D
              XX[i] = X[n, i] - X[n2, i];
              VV[i] = V[n, i] - V[n2, i];
              while (XX[i] > pi)   XX[i] -= 2. * pi;   end
              while (XX[i] < -pi)   XX[i] += 2. * pi;   end
              r2 += XX[i] * XX[i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
            if (close)
              feature_ij = [XX[1], XX[2], XX[3]];
              Π = compute_Π(XX, VV, rho[n], rho[n2], α, β, h, c)
              nn_p_rho = fij_nn(feature_ij)[1]
              tmp = -m * (nn_p_rho + Π) * H(sqrt(r2), h);
              for i in 1 : D
                A[n, i] += tmp * XX[i]
              end


              """
              Compute model derivatives wrt parameters: ∂F/∂θ
              ∂F_p [N, D^2, n_params] (n, 3, :) x-component of F
                n_param = 1,2,3,4   ->   c_in, h, α, β = p_in
              """

              tmpH = H(sqrt(r2), h);

              ∂F_p[n, 4, :] .+= - m * ∂f_∂θk(feature_ij, p_in) * tmpH * XX[1];
              ∂F_p[n, 5, :] .+= - m * ∂f_∂θk(feature_ij, p_in) * tmpH * XX[2];
              ∂F_p[n, 6, :] .+= - m * ∂f_∂θk(feature_ij, p_in) * tmpH * XX[3];

              """
              Compute jacobian: ∂F/∂x
              """
              tmpW = W(sqrt(r2), h);
              tmp∂rH_r = dH_dr_r(sqrt(r2), h);
              if r2 == 0.0
                tmp∂rH_r = 1e6
              end

              ∂Prhonn_∂x = ∂fij_∂x(feature_ij)[1];
              ∂Prhonn_∂y = ∂fij_∂x(feature_ij)[2];
              ∂Prhonn_∂z = ∂fij_∂x(feature_ij)[3];

              ∂Π_∂xi = ∂Π_x(XX, VV, c, h, α, β, rho[n], rho[n2], drhoi_dxi[n, 1], drhoj_dxi[n, 1])
              ∂Π_∂yi = ∂Π_y(XX, VV, c, h, α, β, rho[n], rho[n2], drhoi_dxi[n, 2], drhoj_dxi[n, 2])
              ∂Π_∂zi = ∂Π_z(XX, VV, c, h, α, β, rho[n], rho[n2], drhoi_dxi[n, 3], drhoj_dxi[n, 3])
              ∂Π_∂ui = ∂Π_u(XX, VV, c, h, α, β, rho[n], rho[n2])
              ∂Π_∂vi = ∂Π_v(XX, VV, c, h, α, β, rho[n], rho[n2])
              ∂Π_∂wi = ∂Π_w(XX, VV, c, h, α, β, rho[n], rho[n2]);


              #∂ₓF^x:
              ∂F_x[n, 4, 1] += - (XX[1]*XX[1])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[1]*(∂Π_∂xi + ∂Prhonn_∂x)*tmpH - (m*Π + nn_p_rho)*tmpH
              #∂yF^x:
              ∂F_x[n, 4, 2] += - (XX[1]*XX[2])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[1]*(∂Π_∂yi + ∂Prhonn_∂y)*tmpH
              #∂zF^x:
              ∂F_x[n, 4, 3] += - (XX[1]*XX[3])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[1]*(∂Π_∂zi + ∂Prhonn_∂z)*tmpH
              #∂uF^x:
              ∂F_x[n, 4, 4] += -(m*XX[1]*tmpH)*∂Π_∂ui #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]))
              #∂vF^x:
              ∂F_x[n, 4, 5] += -(m*XX[1]*tmpH)*∂Π_∂vi
              #∂wF^x:
              ∂F_x[n, 4, 6] += -(m*XX[1]*tmpH)*∂Π_∂wi

              #∂ₓF^y:
              ∂F_x[n, 5, 1] += - (XX[2]*XX[1])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[2]*(∂Π_∂xi + ∂Prhonn_∂x)*tmpH
              #∂yF^y:
              ∂F_x[n, 5, 2] += - (XX[2]*XX[2])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[2]*(∂Π_∂yi + ∂Prhonn_∂y)*tmpH - (m*Π + nn_p_rho)*tmpH
              #∂zF^y:
              ∂F_x[n, 5, 3] += - (XX[2]*XX[3])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[2]*(∂Π_∂zi + ∂Prhonn_∂z)*tmpH
              #∂uF^y:
              ∂F_x[n, 5, 4] += -(m*XX[2]*tmpH)*∂Π_∂ui
              #∂vF^y:
              ∂F_x[n, 5, 5] += -(m*XX[2]*tmpH)*∂Π_∂vi #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]))
              #∂wF^y:
              ∂F_x[n, 5, 6] += -(m*XX[2]*tmpH)*∂Π_∂wi

              #∂ₓF^z:
              ∂F_x[n, 6, 1] += - (XX[3]*XX[1])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[3]*(∂Π_∂xi + ∂Prhonn_∂x)*tmpH
              #∂yF^z:
              ∂F_x[n, 6, 2] += - (XX[3]*XX[2])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[3]*(∂Π_∂yi + ∂Prhonn_∂y)*tmpH
              #∂zF^z:
              ∂F_x[n, 6, 3] += - (XX[3]*XX[3])*(m*Π + nn_p_rho)*tmp∂rH_r - m*XX[3]*(∂Π_∂zi + ∂Prhonn_∂z)*tmpH - (m*Π + nn_p_rho)*tmpH
              #∂uF^z:
              ∂F_x[n, 6, 4] += -(m*XX[3]*tmpH)*∂Π_∂ui
              #∂vF^z:
              ∂F_x[n, 6, 5] += -(m*XX[3]*tmpH)*∂Π_∂vi
              #∂wF^z:
              ∂F_x[n, 6, 6] += -(m*XX[3]*tmpH)*∂Π_∂wi #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]))
            end
          end
    end   end   end
  end
  for i in 1 : D
    A[:, i] += (θ/tke) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho, ∂F_x, ∂F_p
end



#---------------------Obtain A half step function


# n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;
function obtain_sph_AV_A(X, V, θ, p_in)
  fij_nn(x) = re(p_in)(x)
  A = zeros(N, D);
  rho = zeros(N);
  mPdrho2 = zeros(N);
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
  XX = zeros(D);
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
              while (XX[i] > pi)   XX[i] -= 2. * pi;   end
              while (XX[i] < -pi)   XX[i] += 2. * pi;   end
              r2 += XX[i] * XX[i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
            if (close)
              tmp = m * W(sqrt(r2), h); rho[n] += tmp;
            end
          end
    end   end   end
  end

  XX = zeros(D); VV = zeros(D);
  # computing A
  for n in 1 : N
    # A[n, :] = obtain_forcing_A(X[n, :]);
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

            close = true;   r2 = 0.;
            for i in 1 : D
              XX[i] = X[n, i] - X[n2, i];
              VV[i] = V[n, i] - V[n2, i];
              while (XX[i] > pi)   XX[i] -= 2. * pi;   end
              while (XX[i] < -pi)   XX[i] += 2. * pi;   end
              r2 += XX[i] * XX[i];
              if (r2 > 4. * h * h)   close = false; break;   end
            end
            if (close)
              feature_ij = [XX[1], XX[2], XX[3]];
              Π = compute_Π(XX, VV, rho[n], rho[n2], α, β, h, c)
              nn_p_rho = fij_nn(feature_ij)[1]
              tmp = -m*(nn_p_rho + Π) * H(sqrt(r2), h);
              for i in 1 : D
                A[n, i] += tmp * XX[i]
              end
            end
          end
    end   end   end
  end
  for i in 1 : D
    A[:, i] += (θ/tke) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho
end
