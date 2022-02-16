

"""
					Mixed mode Senstivity analysis SPH_AV

Rotationally invaraint NN for summand term
  ∑ NN(p/ρ^2, xx * vv, ||xx||)*xx

"""


n_features = 5;
# 
# NN = Chain(
#       Dense(n_features, height, tanh), #R^1 -> R^h
#       Dense(height, height, tanh),
#       Dense(height, 1)        #R^h -> R^1
#     )
#
#
# p_hat, re = Flux.destructure(NN)   #flatten nn params
# n_params = size(p_hat)[1]

f_nn(x) = NN(x)  #input needs to be [float]


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

function ∂P_rhoi2_∂z(rhoi, ∂ρi_∂zi, c, g)
  Pi = Pres(rhoi, c, g)
  ∂Pi_∂z = c^2 * rhoi^(g-1) * ∂ρi_∂zi
  ∂P_ρ2i_∂z = (∂Pi_∂z * rhoi^2 - 2*Pi*rhoi*∂ρi_∂zi)/(rhoi^4)
  return ∂P_ρ2i_∂z
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
              for i in 1 : D
                drhoi_dxi[n, i] += m * H(sqrt(r2), h) * XX[i] #computes ∂ρᵢ/∂xᵢ
                drhoj_dxi[n, i] = m * H(sqrt(r2), h) * XX[i] #computes ∂ρⱼ/∂xᵢ
              end
            end
          end
        end
    end   end
  end

  for n in 1 : N
    mPdrho2[n] = m * P_d_rho2(rho[n], Pres, c_gt, g);
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
              feature_ij = [mPdrho2[n], mPdrho2[n2], XX'*VV, sqrt(r2), h];
              rot_term = fij_nn(feature_ij)[1]
              for i in 1 : D
                A[n, i] += rot_term*XX[i] #+ θ * (V[n, i] - mean(V[:, i]))
              end


              """
              Compute model derivatives wrt parameters: ∂F/∂θ
              ∂F_p [N, D^2, n_params] (n, 3, :) x-component of F
                n_param = 1,2,3,4   ->   c_in, h, α, β = p_in
              """

              ∂f_θ = ∂f_∂θk(feature_ij, p_in);

              ∂F_p[n, 4, :] .+= ∂f_θ*XX[1]
              ∂F_p[n, 5, :] .+= ∂f_θ*XX[2]
              ∂F_p[n, 6, :] .+= ∂f_θ*XX[3]


              """
              Compute jacobian: ∂F/∂x
              """

              ∂Prhoi_∂x = ∂P_rhoi2_∂x(rho[n], drhoi_dxi[n, 1], c, g)
              ∂Prhoj_∂x = ∂P_rhoi2_∂x(rho[n2], drhoj_dxi[n2, 1], c, g)
              ∂Prhoi_∂y = ∂P_rhoi2_∂y(rho[n], drhoi_dxi[n, 2], c, g)
              ∂Prhoj_∂y = ∂P_rhoi2_∂y(rho[n2], drhoi_dxi[n2, 2], c, g)
              ∂Prhoi_∂z = ∂P_rhoi2_∂z(rho[n], drhoi_dxi[n, 2], c, g)
              ∂Prhoj_∂z = ∂P_rhoi2_∂z(rho[n2], drhoi_dxi[n2, 2], c, g)

              ∂fij_X = ∂fij_∂x(feature_ij);
              ∂fij_x1 = ∂fij_X[1]; ∂fij_x2 = ∂fij_X[2];
              ∂fij_x3 = ∂fij_X[3]; ∂fij_x4 = ∂fij_X[4];

              #∂ₓF^x:
              ∂F_x[n, 4, 1] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[1] + ∂fij_x4 * XX[1]/(sqrt(r2) + 1e-4))*XX[1] + rot_term;
              #∂yF^x:
              ∂F_x[n, 4, 2] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[2] + ∂fij_x4 * XX[2]/(sqrt(r2) + 1e-4))*XX[1];
              #∂zF^x:
              ∂F_x[n, 4, 3] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[3] + ∂fij_x4 * XX[3]/(sqrt(r2) + 1e-4))*XX[1];
              #∂uF^x:
              ∂F_x[n, 4, 4] += (∂fij_x3 * XX[1])*XX[1]; #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]));
              #∂vF^x:
              ∂F_x[n, 4, 5] += (∂fij_x3 * XX[2])*XX[1];
              #∂wF^x:
              ∂F_x[n, 4, 6] += (∂fij_x3 * XX[3])*XX[1];

              #∂ₓF^y:
              ∂F_x[n, 5, 1] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[1] + ∂fij_x4 * XX[1]/(sqrt(r2) + 1e-4))*XX[2]
              #∂yF^y:
              ∂F_x[n, 5, 2] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[2] + ∂fij_x4 * XX[2]/(sqrt(r2) + 1e-4))*XX[2] + rot_term;
              #∂zF^y:
              ∂F_x[n, 5, 3] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[3] + ∂fij_x4 * XX[3]/(sqrt(r2) + 1e-4))*XX[2]
              #∂uF^y:
              ∂F_x[n, 5, 4] += (∂fij_x3 * XX[1])*XX[2];
              #∂vF^y:
              ∂F_x[n, 5, 5] += (∂fij_x3 * XX[2])*XX[2];   #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]));
              #∂wF^y:
              ∂F_x[n, 5, 6] += (∂fij_x3 * XX[3])*XX[2];

              #∂ₓF^z:
              ∂F_x[n, 6, 1] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[1] + ∂fij_x4 * XX[1]/(sqrt(r2) + 1e-4))*XX[3];
              #∂yF^z:
              ∂F_x[n, 6, 2] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[2] + ∂fij_x4 * XX[2]/(sqrt(r2) + 1e-4))*XX[3]
              #∂zF^z:
              ∂F_x[n, 6, 3] += (∂fij_x1 * ∂Prhoi_∂x + ∂fij_x2 * ∂Prhoj_∂x + ∂fij_x3 * VV[3] + ∂fij_x4 * XX[3]/(sqrt(r2) + 1e-4))*XX[3] + rot_term;
              #∂uF^z:
              ∂F_x[n, 6, 4] += (∂fij_x3 * XX[1])*XX[3];
              #∂vF^z:
              ∂F_x[n, 6, 5] += (∂fij_x3 * XX[2])*XX[3];   #+θ/tke *(1 - 1/N) - θ/N*V[n, 1]/(tke^2)*(V[n, 1] - mean(V[:, 1]));
              #∂wF^z:
              ∂F_x[n, 6, 6] += (∂fij_x3 * XX[3])*XX[3];
            end
          end
        end
    end   end
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
        end
    end   end
  end

  for n in 1 : N
    mPdrho2[n] = m * P_d_rho2(rho[n], Pres, c_gt, g);
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
              feature_ij = [mPdrho2[n], mPdrho2[n2], XX'*VV, sqrt(r2), h];
              rot_term = fij_nn(feature_ij)[1]
              for i in 1 : D
                A[n, i] += rot_term * XX[i] #+ θ * (V[n, i] - mean(V[:, i]))
              end
            end
          end
        end
    end   end
  end
  for i in 1 : D
    A[:, i] += (θ/tke) * (V[:, i] .- mean(V[:, i]))
  end
  return A, rho
end
