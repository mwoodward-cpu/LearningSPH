
"""
A collectiong of loss functions:
    Particle based (L2)
    Probabilistic based (KL)
    Field based
"""


function W(r, h)
  sigma = 1/(pi*h^3)  #3D normalizing factor
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

# H(r) = (d W / d r) / r
function H(r, h)
  sigma = 1/(pi*h^3)  #3D normalizing factor
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * (-3. + 9. * q / 4.) / h^2);
end



#------------ Cont KL


function trapezoid_int_KL(G, G_pred, xs, xf)
	∫ = 0.0;
	Δx = (xf - xs)/n_int;
	x = xs:Δx:xf
	for i in 2 : n_int
		if (G(x[i - 1]) == 0.0) || (G(x[i]) == 0.0)
			integrand = 0.0
		elseif (G_pred(x[i - 1]) == 0.0) || (G_pred(x[i]) == 0.0)
			integrand = 1e4
		else
			f_l = G(x[i - 1]) * log(G(x[i - 1]) / G_pred(x[i - 1]))
			f_r = G(x[i]) * log(G(x[i]) / G_pred(x[i]))
			integrand = (f_l + f_r)/2 * Δx
		end
		∫ += integrand
	end
	return ∫
end


function trapezoid_int_∂KLi(G, G_pred, h_pred, δu_pr, xs, xf)
	∫ = zeros(N);
	Δx = (xf - xs)/n_int;
	x = xs:Δx:xf
	for i in 2 : n_int
		if (G(x[i - 1]) == 0.0) || (G(x[i]) == 0.0)
			integrand = 0.0
		elseif (G_pred(x[i - 1]) == 0.0) || (G_pred(x[i]) == 0.0)
			integrand = 1e4
		else
			f_l = -1/h_pred * G(x[i-1]) * (1/(N*h_pred) * K_prime.((δu_pr .- x[i-1])/h_pred))/G_pred(x[i-1])
			f_r = -1/h_pred * G(x[i]) * (1/(N*h_pred) * K_prime.((δu_pr .- x[i])/h_pred))/G_pred(x[i])
			integrand = (f_l .+ f_r)/2 * Δx
		end
		∫ .+= integrand
	end
	return ∫
end


function Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, t)
    #Forward KL: data is sampled from GT distritubtion
	#For each τ, this integrates over the variable z in kl(τ, z, z_data)
	L = 0.0;
	δu_gt = Vel_inc_gt[t, :, 1]; δu_pr = Vel_inc_pred[t, :, 1];
	δv_gt = Vel_inc_gt[t, :, 2]; δv_pr = Vel_inc_pred[t, :, 2];
	δw_gt = Vel_inc_gt[t, :, 3]; δw_pr = Vel_inc_pred[t, :, 3];

	G_u(x) = kde(x, δu_gt); hu = obtain_h_kde(δu_gt)
	G_v(x) = kde(x, δv_gt); hv = obtain_h_kde(δv_gt)
	G_w(x) = kde(x, δw_gt); hw = obtain_h_kde(δw_gt)


	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
	Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

	ηu = hu; ηv = hv; ηpu = hu_pred; ηpv = hv_pred;
	ηw = hw; ηpw = hw_pred

	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
	xweg = maximum(δw_gt) + r*ηw;  xwsg = minimum(δw_gt) - r*ηw;

	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;
	xwep = maximum(δw_pr) + r*ηpw; xwsp = minimum(δw_pr) - r*ηpw;

	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
	xws = minimum([xwsg, xwsp]); xwe = maximum([xweg, xwep]);

	L = trapezoid_int_KL(G_u, Gu_pred, xus, xue) + trapezoid_int_KL(G_v, Gv_pred, xvs, xve) +
		trapezoid_int_KL(G_w, Gw_pred, xws, xwe)
    return L
end



function compute_∇L_KL(Vel_inc_gt, Vel_inc_pred, ST)
	∇L = zeros(n_params); τ = T;
	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];
	δw_gt = Vel_inc_gt[τ, :, 3]; δw_pr = Vel_inc_pred[τ, :, 3];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
	Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

	ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;
	ηw = hw_kde_gt; ηpw = hw_pred;

	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
	xweg = maximum(δw_gt) + r*ηw;  xwsg = minimum(δw_gt) - r*ηw;

	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;
	xwep = maximum(δw_pr) + r*ηpw; xwsp = minimum(δw_pr) - r*ηpw;

	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
	xws = minimum([xwsg, xwsp]); xwe = maximum([xweg, xwep]);

	∂Iklu_∂z = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr, xus, xue);
	∂Iklv_∂z = trapezoid_int_∂KLi(G_v, Gv_pred, hv_pred, δv_pr, xvs, xve);
	∂Iklw_∂z = trapezoid_int_∂KLi(G_w, Gw_pred, hw_pred, δw_pr, xws, xwe);
	for i ∈ 1 : N
		∇L .+= ∂Iklu_∂z[i] .* ST[τ+1, i, 4, :] + ∂Iklv_∂z[i] .* ST[τ+1, i, 5, :] + ∂Iklw_∂z[i] .* ST[τ+1, i, 6, :]
	end
	return ∇L
end


function compute_∇L_KL_one(Vel_inc_gt, Vel_inc_pred, ST)
	∇L = zeros(n_params); τ = T;
	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	ηu = hu_kde_gt; ηpu = hu_pred;
	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	∂Iklu_∂z = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr, xus, xue);
	for i ∈ 1 : N
		∇L .+= ∂Iklu_∂z[i] .* ST[τ+1, i, 4, :]
	end
	return ∇L
end

function Ikl_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu, hv)
	τ = T; L = 0.0;
	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	ηu = hu; ηpu = hu_pred;
	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	L = trapezoid_int_KL(G_u, Gu_pred, xus, xue)
    return L
end




function Ikl_τ(Vel_inc_gt, Vel_inc_pred, t)
    #Forward KL: data is sampled from GT distritubtion
	#For each τ, this integrates over the variable z in kl(τ, z, z_data)
	L = 0.0;
	for τ in 1 : t
		δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
		δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];
		δw_gt = Vel_inc_gt[τ, :, 3]; δw_pr = Vel_inc_pred[τ, :, 3];

		Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
		Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
		Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

		Gu(x) = Gt_u(x, τ); Gv(x) = Gt_v(x, τ); Gw(x) = Gt_w(x, τ);

		ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;
		ηw = hw_kde_gt; ηpw = hw_pred;

		xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
		xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
		xweg = maximum(δw_gt) + r*ηw;  xwsg = minimum(δw_gt) - r*ηw;

		xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
		xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;
		xwep = maximum(δw_pr) + r*ηpw; xwsp = minimum(δw_pr) - r*ηpw;

		xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
		xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
		xws = minimum([xwsg, xwsp]); xwe = maximum([xweg, xwep]);

		L += dt * (trapezoid_int_KL(Gu, Gu_pred, xus, xue) + trapezoid_int_KL(Gv, Gv_pred, xvs, xve) +
					trapezoid_int_KL(Gw, Gw_pred, xws, xwe))
	end
	return L
end



function compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, ST)
	∇L = zeros(n_params);
	for τ in 1 : T
		δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
		δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];
		δw_gt = Vel_inc_gt[τ, :, 3]; δw_pr = Vel_inc_pred[τ, :, 3];

		Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
		Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
		Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

		Gu(x) = Gt_u(x, τ); Gv(x) = Gt_v(x, τ); Gw(x) = Gt_w(x, τ);

		ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;
		ηw = hw_kde_gt; ηpw = hw_pred;

		xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
		xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
		xweg = maximum(δw_gt) + r*ηw;  xwsg = minimum(δw_gt) - r*ηw;

		xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
		xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;
		xwep = maximum(δw_pr) + r*ηpw; xwsp = minimum(δw_pr) - r*ηpw;

		xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
		xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
		xws = minimum([xwsg, xwsp]); xwe = maximum([xweg, xwep]);

		∂Iklu_∂z = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr, xus, xue);
		∂Iklv_∂z = trapezoid_int_∂KLi(G_v, Gv_pred, hv_pred, δv_pr, xvs, xve);
		∂Iklw_∂z = trapezoid_int_∂KLi(G_w, Gw_pred, hw_pred, δw_pr, xws, xwe);
		for i ∈ 1 : N
			∇L .+= ∂Iklu_∂z[i] .* ST[τ+1, i, 4, :] + ∂Iklv_∂z[i] .* ST[τ+1, i, 5, :] + ∂Iklw_∂z[i] .* ST[τ+1, i, 6, :]
		end
	end
	return ∇L
end


function ∂kl_∂z(Vel_inc_gt, Vel_inc_pred, k)
	∂kl_∂x = zeros(N, 2*D);

	δu_gt = Vel_inc_gt[k, :, 1]; δu_pr = Vel_inc_pred[:, 1];
	δv_gt = Vel_inc_gt[k, :, 2]; δv_pr = Vel_inc_pred[:, 2];
	δw_gt = Vel_inc_gt[k, :, 3]; δw_pr = Vel_inc_pred[:, 3];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
	Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

	Gu(x) = Gt_u(x, k); Gv(x) = Gt_v(x, k); Gw(x) = Gt_w(x, k);

	ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;
	ηw = hw_kde_gt; ηpw = hw_pred;

	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
	xweg = maximum(δw_gt) + r*ηw;  xwsg = minimum(δw_gt) - r*ηw;

	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;
	xwep = maximum(δw_pr) + r*ηpw; xwsp = minimum(δw_pr) - r*ηpw;

	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
	xws = minimum([xwsg, xwsp]); xwe = maximum([xweg, xwep]);

	∂Iklu_∂z = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr, xus, xue);
	∂Iklv_∂z = trapezoid_int_∂KLi(G_v, Gv_pred, hv_pred, δv_pr, xvs, xve);
	∂Iklw_∂z = trapezoid_int_∂KLi(G_w, Gw_pred, hw_pred, δw_pr, xws, xwe);

	for n in 1 : N
		∂kl_∂x[n, 4] = ∂Iklu_∂z[n]
		∂kl_∂x[n, 5] = ∂Iklv_∂z[n]
		∂kl_∂x[n, 6] = ∂Iklw_∂z[n]
	end
	return ∂kl_∂x
end




"""

=================== DISCRETE KL ===========================

"""


function disc_kl_divergence(Vel_inc_gt, Vel_inc_pred)
    #Forward KL: data is sampled from GT distritubtion
    L = 0.0
    M = size(Diff_gt)[2]
	Gu_pred(δu) = kde(δu, Vel_inc_pred[T, :, 1])
	Gv_pred(δv) = kde(δv, Vel_inc_pred[T, :, 2])
	Gw_pred(δw) = kde(δw, Vel_inc_pred[T, :, 3])
        for i in 1 : M
			Gu_i = G_u(Vel_inc_gt[T, i, 1]);
			Gu_predi = Gu_pred(Vel_inc_gt[T, i, 1]);
			Gv_i = G_v(Vel_inc_gt[T, i, 2]);
			Gv_predi = Gv_pred(Vel_inc_gt[T, i, 2]);
			Gw_i = G_w(Vel_inc_gt[T, i, 3]);
			Gw_predi = Gw_pred(Vel_inc_gt[T, i, 3]);
			lu_i = dt/(M*T) * (Gu_i * log(Gu_i / Gu_predi))
			lv_i = dt/(M*T) * (Gv_i * log(Gv_i / Gv_predi))
			lw_i = dt/(M*T) * (Gw_i * log(Gw_i / Gw_predi))
            L += lu_i + lv_i + lw_i
        end
    return L
end


function ∂klu_∂d(Vel_inc_pred, δu)
    #Obtain derivative of kl wrt input δu using AD: forward mode
    Gu_pred(δu) = kde(δu, Vel_inc_pred[T, :, 1])
	if G_u(δu) == 0.0 || Gu_pred(δu) == 0.0
		return 0.0
	end
	kl_ui(δu) = G_u(δu) * log(G_u(δu) / Gu_pred(δu))
    return ForwardDiff.derivative(δu -> kl_ui(δu), δu)
end

function ∂klv_∂d(Vel_inc_pred, δu)
    #Obtain derivative of kl wrt input δv using AD: forward mode
    Gv_pred(δu) = kde(δu, Vel_inc_pred[T, :, 2])
	if G_v(δu) == 0.0 || Gv_pred(δu) == 0.0
		return 0.0
	end
	kl_vi(δu) = G_v(δu) * log(G_v(δu) / Gv_pred(δu))
    return ForwardDiff.derivative(δu -> kl_vi(δu), δu)
end

function ∂klw_∂d(Vel_inc_pred, δu)
    #Obtain derivative of kl wrt input δv using AD: forward mode
    Gw_pred(δu) = kde(δu, Vel_inc_pred[T, :, 3])
	if G_w(δu) == 0.0 || Gw_pred(δu) == 0.0
		return 0.0
	end
	kl_wi(δu) = G_w(δu) * log(G_w(δu) / Gw_pred(δu))
    return ForwardDiff.derivative(δu -> kl_wi(δu), δu)
end


function ∂kl_∂x_i(Vel_inc_gt, Vel_inc_pred, i)
	"""
	returns gradient of summand term in KL wrt x,y,u,v
	"""
	δu_i = Vel_inc_gt[T, i, 1]; δv_i = Vel_inc_gt[T, i, 2];
	δw_i = Vel_inc_gt[T, i, 3];
	∂δui_∂ui = 1.0; ∂δvi_∂vi = 1.0; ∂δwi_∂wi = 1.0;

	L_x = 0.0											# ∂L/∂x(τ)_i
	L_y = 0.0									  	 	# ∂L/∂y(τ)_i
	L_z = 0.0
	L_u = -∂klu_∂d(Vel_inc_pred, δu_i) * ∂δui_∂ui		# ∂L/∂u(τ)_i
	L_v = -∂klv_∂d(Vel_inc_pred, δv_i) * ∂δvi_∂vi		# ∂L/∂v(τ)_i
	L_w = -∂klw_∂d(Vel_inc_pred, δw_i) * ∂δwi_∂wi		# ∂L/∂v(τ)_i
	return [L_x, L_y, L_z, L_u, L_v, L_w]
end

function ∂kl2_∂x_i(Vel_inc_gt, Vel_inc_pred, i)
	"""
	returns gradient of summand term in KL wrt x,y,u,v
	"""
	δu_i = Vel_inc_gt[T, i, 1]; δv_i = Vel_inc_gt[T, i, 2];
	δw_i = Vel_inc_gt[T, i, 3];
	∂δui_∂ui = 1.0; ∂δvi_∂vi = 1.0; ∂δwi_∂wi = 1.0;

	δu_gt = Vel_inc_gt[T, :, 1]; δu_pr = Vel_inc_pred[T, :, 1];
	δv_gt = Vel_inc_gt[T, :, 2]; δv_pr = Vel_inc_pred[T, :, 2];
	δw_gt = Vel_inc_gt[T, :, 3]; δw_pr = Vel_inc_pred[T, :, 3];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
	Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

	function ∂klu_∂zpi(x, i)
		return -1/hu_pred * G_u(x) * (1/(N*hu_pred) * K_prime((δu_pr[i] - x)/hu_pred))/Gu_pred(x)
	end
	function ∂klv_∂zpi(x, i)
		return -1/hv_pred * G_v(x) * (1/(N*hv_pred) * K_prime((δv_pr[i] - x)/hv_pred))/Gv_pred(x)
	end
	function ∂klw_∂zpi(x, i)
		return -1/hw_pred * G_w(x) * (1/(N*hw_pred) * K_prime((δw_pr[i] - x)/hw_pred))/Gw_pred(x)
	end

	L_x = 0.0											# ∂L/∂x(τ)_i
	L_y = 0.0									  	 	# ∂L/∂y(τ)_i
	L_z = 0.0
	L_u = ∂klu_∂zpi(δu_i, i) * ∂δui_∂ui		# ∂L/∂u(τ)_i
	L_v = ∂klv_∂zpi(δv_i, i) * ∂δvi_∂vi		# ∂L/∂v(τ)_i
	L_w = ∂klw_∂zpi(δw_i, i) * ∂δwi_∂wi		# ∂L/∂v(τ)_i
	return [L_x, L_y, L_z, L_u, L_v, L_w]
end


function disc_compute_∇L(Vel_inc_gt, Vel_inc_pred, ST)
	∇L = zeros(n_params)
		for i ∈ 1 : N
			∂L_x = ∂kl2_∂x_i(Vel_inc_gt, Vel_inc_pred, i)
			for k in 1 : n_params
				∇L[k] += ∂L_x'*ST[T, i, :, k]
			end
			# ∇L[1] += ∂L_x'*ST[τ, i, :, 1]
			# ∇L[2] = 0.0
			# ∇L[k] += ∂L_x[1] * ST[τ, i, 1, k] + ∂L_x[2] * ST[τ, i, 2, k] + ∂L_x[3] * ST[τ, i, 3, k] + ∂L_x[4] * ST[τ, i, 4, k]
		end
	if isnan(∇L[1]) || isinf(∇L[1])
		∇L = zeros(n_params);
	end
	return ∇L
end







"""


=================== L2 ======================
					&
================== MSE ======================

"""


function ∂xLᵢ(X, X_gt, V, V_gt, i)
	xdif = X[i, 1] - X_gt[i, 1]
	ydif = X[i, 2] - X_gt[i, 2]
	zdif = X[i, 3] - X_gt[i, 3]
	udif = V[i, 1] - V_gt[i, 1]
	vdif = V[i, 2] - V_gt[i, 2]
	wdif = V[i, 3] - V_gt[i, 3]

	L_x = (xdif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂x_i
	L_y = (ydif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂y_i
	L_z = (zdif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂y_i
	L_u = (udif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂u_i
	L_v = (vdif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂v_i
	L_w = (wdif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂v_i
	return [L_x, L_y, L_z, L_u, L_v, L_w]
end


function ∂xmseᵢ(X, X_gt, V, V_gt, i)
	"""
	deriv of MSE
	"""
	xdif = X[i, 1] - X_gt[i, 1]
	ydif = X[i, 2] - X_gt[i, 2]
	zdif = X[i, 3] - X_gt[i, 3]
	udif = V[i, 1] - V_gt[i, 1]
	vdif = V[i, 2] - V_gt[i, 2]
	wdif = V[i, 3] - V_gt[i, 3]
	L_x = 2*(xdif)/(N*T) # ∂L/∂x_i
	L_y = 2*(ydif)/(N*T) # ∂L/∂y_i
	L_z = 2*(zdif)/(N*T) # ∂L/∂z_i
	L_u = 2*(udif)/(N*T) # ∂L/∂u_i
	L_v = 2*(vdif)/(N*T) # ∂L/∂v_i
	L_w = 2*(wdif)/(N*T) # ∂L/∂w_i
	return [L_x, L_y, L_z, L_u, L_v, L_w]
end


function compute_∂L_∂θ_τ(traj_pred, traj_gt, vel_pred, vel_gt, ST)
	# global H_λ, H_U
	"""
	sums over t and i and compute the gradient wrt each param
	only learning c
	"""
	∇L = zeros(n_params)
	for t ∈ 2 : (T+1)
		for i ∈ 1 : N
			# L_x = ∂xLᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
			∂L_x = ∂xmseᵢ(traj_pred[t,:,:], traj_gt[t,:,:], vel_pred[t,:,:], vel_gt[t,:,:], i)
			# for k ∈ 1 : n_params
			# 	∇L[k] += ∂L_x' * ST[τ, i, :, k]
			# end
			∇L[1] += ∂L_x'*ST[t, i, :, 1]
			∇L[2] = 0.0 #dummy variable
		end
	end
	return ∇L
end


function compute_∂L_mse_k(traj_pred, traj_gt, vel_pred, vel_gt, ST)
	# global H_λ, H_U
	"""
	sums over t and i and compute the gradient wrt each param
	"""
	∇L = zeros(n_params)
	for t ∈ 2 : (T+1)
		for i ∈ 1 : N
			# L_x = ∂xLᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
			∂L_x = ∂xmseᵢ(traj_pred[t,:,:], traj_gt[t,:,:], vel_pred[t,:,:], vel_gt[t,:,:], i)
			for k ∈ 1 : n_params
				∇L[k] += ∂L_x' * ST[t, i, :, k]
			end
		end
	end
	return ∇L
end



function compute_∂L_∂θ_fixedτ(traj_pred, traj_gt, vel_pred, vel_gt, ST)
	# global H_λ, H_U
	"""
	sums over t and i and compute the gradient wrt each param
	"""
	∇L = zeros(n_params)
	for i ∈ 1 : N
		# L_x = ∂xLᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
		∂L_x = ∂xmseᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
		for k ∈ 1 : n_params
			∇L[k] += ∂L_x' * ST[τ+1, i, :, k]
		end
	end
	return ∇L
end






"""


=================== Field ======================
					based
=================== Loss ======================

TODO:
	need rho over time from gt and pred sims
	try over T steps

"""

#----------GRID (uniform)

D = 3;
gsp = 4; #produces grid of 2^gsp x 2^gsp x 2^gsp number of particles
pgrid = Iterators.product((2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi)
pgrid = vec(collect.(pgrid)) #particles

X_grid = zeros(N, D);
for n in 1 : N
  X_grid[n, 1] = pgrid[n][1]
  X_grid[n, 2] = pgrid[n][2]
  X_grid[n, 3] = pgrid[n][3]
end

Plots.scatter(X_grid[:, 1], X_grid[:, 2], X_grid[:, 3])


n_hash = floor(Int, 2*pi / h);   l_hash = 2*pi / n_hash;
function obtain_interpolated_velocity(X_grid, X, V, rho)
  Vf = zeros(N,D);
  ∂Vf_∂x = zeros(N, D); ∂Vf_∂y = zeros(N, D); ∂Vf_∂z = zeros(N, D);
  ∂Vf_∂u = zeros(N, D); ∂Vf_∂v = zeros(N, D); ∂Vf_∂w = zeros(N, D);
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
    x_hash = [floor(Int, X_grid[n, 1] / l_hash) + 1,
  			floor(Int, X_grid[n, 2] / l_hash) + 1,
  			floor(Int, X_grid[n, 3] / l_hash) + 1];
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
            XX[i] = X_grid[n, i] - X[n2, i];
            while (XX[i] > pi)   XX[i] -= 2. * pi;   end
            while (XX[i] < -pi)   XX[i] += 2. * pi;   end
            r2 += XX[i] * XX[i];
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)

			for i in 1 : D
	            Vf[n, i] += m * W(sqrt(r2), h) * V[n2, i] / rho[n2];
				∂Vf_∂x[n, i] += m * H(sqrt(r2), h) * V[n2, i] * XX[1] / rho[n2];
				∂Vf_∂y[n, i] += m * H(sqrt(r2), h) * V[n2, i] * XX[2] / rho[n2];
				∂Vf_∂z[n, i] += m * H(sqrt(r2), h) * V[n2, i] * XX[3] / rho[n2];
			end
			∂Vf_∂u[n, 1] = m * W(0.0, h) / rho[n2];
			∂Vf_∂v[n, 2] = m * W(0.0, h) / rho[n2];
			∂Vf_∂w[n, 3] = m * W(0.0, h) / rho[n2];
          end
        end
    end   end   end
  end
  return Vf, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w
end


function obtain_interpolated_velocity_over_τ(traj, vels, rhos, T)
	Vf_t = zeros(T+1,N,D);
	∂Vft_∂x = zeros(T+1,N,D); ∂Vft_∂y = zeros(T+1,N,D); ∂Vft_∂z = zeros(T+1,N,D);
	∂Vft_∂u = zeros(T+1,N,D); ∂Vft_∂v = zeros(T+1,N,D); ∂Vft_∂w = zeros(T+1,N,D);
	for t in 1 : (T+1)
		Vf_t[t,:,:], ∂Vft_∂x[t,:,:], ∂Vft_∂y[t,:,:], ∂Vft_∂z[t,:,:], ∂Vft_∂u[t,:,:], ∂Vft_∂v[t,:,:], ∂Vft_∂w[t,:,:] =
		obtain_interpolated_velocity(X_grid, traj[t,:,:], vels[t,:,:], rhos[t,:])
	end
	return Vf_t, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂z, ∂Vft_∂u, ∂Vft_∂v, ∂Vft_∂w
end




function ∂L_∂Y(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w, t, n)
	∂L_∂xi = 0.0; ∂L_∂yi = 0.0; ∂L_∂zi = 0.0;
	∂L_∂ui = 0.0; ∂L_∂vi = 0.0; ∂L_∂wi = 0.0;

	for i in 1 : D
		∂L_∂xi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂x[t, n, i])
		∂L_∂yi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂y[t, n, i])
		∂L_∂zi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂z[t, n, i])
		∂L_∂ui += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂u[t, n, i])
		∂L_∂vi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂v[t, n, i])
		∂L_∂wi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂w[t, n, i])
	end
	return [∂L_∂xi, ∂L_∂yi, ∂L_∂zi, ∂L_∂ui, ∂L_∂vi, ∂L_∂wi]
end


function compute_∇L_field(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w, ST)
	∇L = zeros(n_params)
	for t in 2 : (T + 1)
		for n ∈ 1 : N
			∂L_x = ∂L_∂Y(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w, t, n)
			for k ∈ 1 : n_params
				∇L[k] += ∂L_x' * ST[t, n, :, k]
			end
		end
	end
	return ∇L
end







function compute_Lg(lg_method, Vel_inc_gt, Vel_inc_pred, traj_pred, vels_pred, rhos_pred, Vfp, Vf_gt, t)
	Lg = 0.0;
	if lg_method == "kl_t"
		Lg = Ikl_τ(Vel_inc_gt, Vel_inc_pred, t)
	end
	if lg_method =="kl_lf"
		Lg_kl = Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, t)
		# Vfp, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂z, ∂Vft_∂u, ∂Vft_∂v, ∂Vft_∂w =
		# obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, t)
		Lg_lf = mse(Vfp, Vf_gt);
		Lg = Lg_kl + Lg_lf;
	end
	if lg_method =="lf"
		Lg = mse(Vfp, Vf_gt);
	end
	return Lg
end

function compute_L_comp(loss_method, Vel_inc_gt, Vel_inc_pred, traj_pred, vels_pred, rhos_pred, Vfp, Vf_gt, t)
	L = 0.0;
	if loss_method == "kl_lf"
		L_kl = Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, t)
		# Vfp, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂z, ∂Vft_∂u, ∂Vft_∂v, ∂Vft_∂w =
		# obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, t)
		L_lf = mse(Vfp, Vf_gt);
		L = L_kl + L_lf;
	end
	return L
end







function compute_L(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt,
	 			  traj_gt, traj_pred, vels_pred, rhos_pred)
	L = 0.0;
	if loss_method =="l2"
		L = mse(traj_gt[1:(T+1),:,:], traj_pred);
	end
	if loss_method == "kl"
		L = Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, G_w, hu_kde_gt, hv_kde_gt, hw_kde_gt)
	end
	if loss_method =="kl_lf"
		L_kl = Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, G_w, hu_kde_gt, hv_kde_gt, hw_kde_gt)
		Vfp, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂z, ∂Vft_∂u, ∂Vft_∂v, ∂Vft_∂w =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		L_lf = mse(Vfp, Vf_gt);
		L = L_kl + L_lf;
	end
	if loss_method == "kl_t"
		L = Ikl_τ(Vel_inc_gt, Vel_inc_pred)
	end
	if loss_method == "kl_t_one_dist"
		L = Ikl_τ_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt)
	end
	if loss_method == "kl_one_dist"
		L = Ikl_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt)
	end
	if loss_method == "kl_l2_t"
		Lkl = Ikl_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt)
		Ll2 = mse(traj_gt[1:(T+1),:,:], traj_pred);
		L = Lkl + Ll2;
	end
	if loss_method == "kl_lf_t"
		Vfp, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂z, ∂Vft_∂u, ∂Vft_∂v, ∂Vft_∂w =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		Lkl = Ikl_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt)
		Lf = mse(Vfp, Vf_gt);
		L = Lkl + Lf;
	end
	if loss_method == "dkl_t"
		L = disc_kl_divergence_t(Vel_inc_gt, Vel_inc_pred)
	end
	if loss_method =="lf"
		Vfp, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂z, ∂Vft_∂u, ∂Vft_∂v, ∂Vft_∂w =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		L = mse(Vfp, Vf_gt);
	end
	if isinf(L) L = 1e2; end
	return L
end

function compute_∇L(loss_method, Vel_inc_gt, Vel_inc_pred, traj_pred, traj_gt,
	vels_pred, vels_gt, rhos_pred, Vf_gt, ST)
	if loss_method == "l2"
		∇L = compute_∂L_mse_k(traj_pred, traj_gt, vels_pred, vels_gt, ST)
	end
	if loss_method == "kl"
		∇L = compute_∇L_KL(Vel_inc_gt, Vel_inc_pred, ST)
	end
	if loss_method == "kl_lf"
		∇L_kl = compute_∇L_KL(Vel_inc_gt, Vel_inc_pred, ST)
		Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		∇L_lf = compute_∇L_field(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w, ST)
		∇L = ∇L_kl + ∇L_lf
	end
	if loss_method == "kl_t"
		∇L = compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, ST)
	end
	if loss_method == "kl_t_one_dist"
		∇L = compute_∇L_KL_τ_one(Vel_inc_gt, Vel_inc_pred, ST)
	end
	if loss_method == "kl_one_dist"
		∇L = compute_∇L_KL_one(Vel_inc_gt, Vel_inc_pred, ST)
	end
	if loss_method == "kl_l2_t"
		∇L_kl = compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, ST)
		∇L_l2 = compute_∂L_mse_k(traj_pred, traj_gt, vels_pred, vels_gt, ST)
		∇L = ∇L_kl + ∇L_l2
	end
	if loss_method =="kl_lf_t"
		Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		Diff_pred, Vel_inc_pred = obtain_pred_dists(traj_pred, vels_pred, traj_gt[1,:,:], vels_gt[1,:,:])
		∇L_kl = compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, ST)
		∇L_lf = compute_∇L_field(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w, ST)
		∇L = ∇L_kl + ∇L_lf
	end
	if loss_method == "dkl_t"
		∇L = discKL_compute_∇L_t(Vel_inc_gt, Vel_inc_pred, ST)
	end
	if loss_method =="lf"
		Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T);
		∇L = compute_∇L_field(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂z, ∂Vf_∂u, ∂Vf_∂v, ∂Vf_∂w, ST)
	end
	return ∇L
end
