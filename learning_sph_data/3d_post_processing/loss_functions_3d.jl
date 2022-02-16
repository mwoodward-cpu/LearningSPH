
"""
A collectiong of loss functions:
    Particle based (L2)
    Probabilistic based (KL)
    Field based
"""



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


function trapezoid_int_∂KLi(G, G_pred, h_pred, δu_pri, xs, xf)
	∫ = 0.0;
	Δx = (xf - xs)/n_int;
	x = xs:Δx:xf
	for i in 2 : n_int
		if (G(x[i - 1]) == 0.0) || (G(x[i]) == 0.0)
			integrand = 0.0
		elseif (G_pred(x[i - 1]) == 0.0) || (G_pred(x[i]) == 0.0)
			integrand = 1e4
		else
			f_l = -1/h_pred * G(x[i-1]) * (1/(N*h_pred) * K_prime((δu_pri - x[i-1])/h_pred))/G_pred(x[i-1])
			f_r = -1/h_pred * G(x[i]) * (1/(N*h_pred) * K_prime((δu_pri - x[i])/h_pred))/G_pred(x[i])
			integrand = (f_l + f_r)/2 * Δx
		end
		∫ += integrand
	end
	return ∫
end

function Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, hu, hv, Gt_u, Gt_v, T)
    #Forward KL: data is sampled from GT distritubtion
	#For each τ, this integrates over the variable z in kl(τ, z, z_data)
	τ = T; L = 0.0;
	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);

	G_u(x) = Gt_u(x, t); G_v(x) = Gt_v(x, t);

	ηu = hu; ηv = hv; ηpu = hu_pred; ηpv = hv_pred;

	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;

	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);

	L = trapezoid_int_KL(G_u, Gu_pred, xus, xue) + trapezoid_int_KL(G_v, Gv_pred, xvs, xve)
    return L
end


function compute_∇L_KL(Vel_inc_gt, Vel_inc_pred, HT)
	∇L = zeros(n_params); τ = T;
	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);

	ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;

	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;

	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);

	for i ∈ 1 : N
		∂Iklu_∂zi = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr[i], xus, xue);
		∂Iklv_∂zi = trapezoid_int_∂KLi(G_v, Gv_pred, hv_pred, δv_pr[i], xvs, xve);
		for k in 1 : n_params
			∇L[k] += ∂Iklu_∂zi * HT[τ+1, i, 3, k] + ∂Iklv_∂zi * HT[τ+1, i, 4, k]
		end
	end
	return ∇L
end

function compute_∇L_KL_one(Vel_inc_gt, Vel_inc_pred, HT)
	∇L = zeros(n_params); τ = T;
	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	ηu = hu_kde_gt; ηpu = hu_pred;
	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	for i ∈ 1 : N
		∂Iklu_∂zi = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr[i], xus, xue);
		for k in 1 : n_params
			∇L[k] += ∂Iklu_∂zi * HT[τ+1, i, 3, k]
		end
	end
	return ∇L
end


function Ikl_τ(Vel_inc_gt, Vel_inc_pred, hu, hv, Gt_u, Gt_v, T)
    #Forward KL: data is sampled from GT distritubtion
	#For each τ, this integrates over the variable z in kl(τ, z, z_data)
	L = 0.0;
	for τ in 1 : T
		δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
		δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];

		Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
		Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
		Gu(x) = Gt_u(x, τ); Gv(x) = Gt_v(x, τ);

		ηu = hu(τ); ηv = hv(τ); ηpu = hu_pred; ηpv = hv_pred;

		xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
		xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
		xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
		xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;

		xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
		xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);

		L += dt * (trapezoid_int_KL(Gu, Gu_pred, xus, xue) + trapezoid_int_KL(Gv, Gv_pred, xvs, xve))
	end
	return L
end


function Ikl_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu, hv, T)
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

function Ikl_τ_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu, hv, Gt_u, T)
    #Forward KL: data is sampled from GT distritubtion
	#For each τ, this integrates over the variable z in kl(τ, z, z_data)
	L = 0.0;
	for τ in 1 : T
		δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
		Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
		Gu(x) = Gt_u(x, τ);
		ηu = hu; ηpu = hu_pred;

		xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
		xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;

		xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);

		L += dt * (trapezoid_int_KL(Gu, Gu_pred, xus, xue))
	end
	return L
end



function compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, HT)
	∇L = zeros(n_params);
	for τ in 1 : T
		δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
		δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];

		Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
		Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
		Gu(x) = Gt_u(x, τ); Gv(x) = Gt_u(x, τ);

		ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;

		xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
		xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
		xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
		xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;

		xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
		xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
		for i ∈ 1 : N
			∂Iklu_∂zi = trapezoid_int_∂KLi(Gu, Gu_pred, hu_pred, δu_pr[i], xus, xue);
			∂Iklv_∂zi = trapezoid_int_∂KLi(Gv, Gv_pred, hv_pred, δv_pr[i], xvs, xve);
			for k in 1 : n_params
				∇L[k] += dt * (∂Iklu_∂zi * HT[τ+1, i, 3, k] + ∂Iklv_∂zi * HT[τ+1, i, 4, k])
			end
		end
	end
	return ∇L
end



function compute_∇L_KL_τ_one(Vel_inc_gt, Vel_inc_pred, HT)
	∇L = zeros(n_params);
	for τ in 1 : T
		δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
		Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
		Gu(x) = Gt_u(x, τ);

		ηu = hu_kde_gt; ηpu = hu_pred;

		xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
		xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;

		xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
		for i ∈ 1 : N
			∂Iklu_∂zi = trapezoid_int_∂KLi(Gu, Gu_pred, hu_pred, δu_pr[i], xus, xue);
			for k in 1 : n_params
				∇L[k] += dt * ∂Iklu_∂zi * HT[τ+1, i, 3, k]
			end
		end
	end
	return ∇L
end







function ∂kl_∂z(Vel_inc_gt, Vel_inc_pred, k)
	∂kl_∂x = zeros(N, 2*D);

	δu_gt = Vel_inc_gt[k, :, 1]; δu_pr = Vel_inc_pred[:, 1];
	δv_gt = Vel_inc_gt[k, :, 2]; δv_pr = Vel_inc_pred[:, 2];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);

	ηu = hu_kde_gt; ηv = hv_kde_gt; ηpu = hu_pred; ηpv = hv_pred;

	xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
	xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
	xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
	xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;

	xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
	xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);

	for n in 1 : N
		∂Iklu_∂zi = trapezoid_int_∂KLi(G_u, Gu_pred, hu_pred, δu_pr[n], xus, xue);
		∂Iklv_∂zi = trapezoid_int_∂KLi(G_v, Gv_pred, hv_pred, δv_pr[n], xvs, xve);
		∂kl_∂x[n, 3] = ∂Iklu_∂zi
		∂kl_∂x[n, 4] = ∂Iklv_∂zi
	end
	return ∂kl_∂x
end





"""

=================== DISCRETE KL ===========================

"""


#-------------Discrete KL

function disc_kl_divergence(Vel_inc_gt, Vel_inc_pred)
    #Forward KL: data is sampled from GT distritubtion
    L = 0.0
    M = size(Diff_gt)[2]
	Gu_pred(δu) = kde(δu, Vel_inc_pred[T, :, 1])
	Gv_pred(δv) = kde(δv, Vel_inc_pred[T, :, 2])
	# Gw_pred(δw) = kde(δw, Vel_inc_pred[T, :, 3])
        for i in 1 : M
			Gu_i = G_u(Vel_inc_gt[T, i, 1]);
			Gu_predi = Gu_pred(Vel_inc_gt[T, i, 1]);
			Gv_i = G_v(Vel_inc_gt[T, i, 2]);
			Gv_predi = Gv_pred(Vel_inc_gt[T, i, 2]);
			# Gw_i = G_w(Vel_inc_gt[T, i, 3]);
			# Gw_predi = Gw_pred(Vel_inc_gt[T, i, 3]);
			lu_i = dt/(M*T) * (Gu_i * log(Gu_i / Gu_predi))
			lv_i = dt/(M*T) * (Gv_i * log(Gv_i / Gv_predi))
			# lw_i = dt/(M*T) * (Gw_i * log(Gw_i / Gw_predi))
            L += lu_i + lv_i #+ lw_i
        end
    return L
end

function disc_kl_divergence_t(Vel_inc_gt, Vel_inc_pred)
    #Forward KL: data is sampled from GT distritubtion
    L = 0.0
    # M = size(Diff_gt)[2]
	Gu_pred(δu, τ) = kde(δu, Vel_inc_pred[τ, :, 1])
	Gv_pred(δv, τ) = kde(δv, Vel_inc_pred[τ, :, 2])

	for τ in 1 : T
        for i in 1 : N
			Gu_i = Gt_u(Vel_inc_gt[τ, i, 1], τ);
			Gu_predi = Gu_pred(Vel_inc_gt[τ, i, 1], τ);
			Gv_i = Gt_v(Vel_inc_gt[τ, i, 2], τ);
			Gv_predi = Gv_pred(Vel_inc_gt[τ, i, 2], τ);
			lu_i = dt/(M*T) * (Gu_i * log(Gu_i / Gu_predi))
			lv_i = dt/(M*T) * (Gv_i * log(Gv_i / Gv_predi))
            L += lu_i + lv_i
        end
	end
    return L
end


function ∂klu_∂d(Vel_inc_pred, δu, τ)
    #Obtain derivative of kl wrt input δu using AD: forward mode
    Gu_pred(δu) = kde(δu, Vel_inc_pred[τ, :, 1])
	if G_u(δu) == 0.0 || Gu_pred(δu) == 0.0
		return 0.0
	end
	kl_ui(δu) = G_u(δu) * log(G_u(δu) / Gu_pred(δu))
    return ForwardDiff.derivative(δu -> kl_ui(δu), δu)
end

function ∂klv_∂d(Vel_inc_pred, δv, τ)
    #Obtain derivative of kl wrt input δv using AD: forward mode
    Gv_pred(δv) = kde(δv, Vel_inc_pred[τ, :, 2])
	if G_v(δv) == 0.0 || Gv_pred(δv) == 0.0
		return 0.0
	end
	kl_vi(δv) = G_v(δv) * log(G_v(δv) / Gv_pred(δv))
    return ForwardDiff.derivative(δv -> kl_vi(δv), δv)
end

function ∂kl_∂x_i(Vel_inc_gt, Vel_inc_pred, i)
	"""
	returns gradient of summand term in KL wrt x,y,u,v
	"""
	δu_i = Vel_inc_gt[T, i, 1]; δv_i = Vel_inc_gt[T, i, 2];
	∂δui_∂ui = 1.0; ∂δvi_∂vi = 1.0;

	L_x = 0.0											# ∂L/∂x(τ)_i
	L_y = 0.0									  	 	# ∂L/∂y(τ)_i
	L_u = -∂klu_∂d(Vel_inc_pred, δu_i, T) * ∂δui_∂ui		# ∂L/∂u(τ)_i
	L_v = -∂klv_∂d(Vel_inc_pred, δv_i, T) * ∂δvi_∂vi		# ∂L/∂v(τ)_i
	return [L_x, L_y, L_u, L_v]
end


function ∂kl_∂x_i_t(Vel_inc_gt, Vel_inc_pred, i, τ)
	"""
	returns gradient of summand term in KL wrt x,y,u,v
	"""
	δu_i = Vel_inc_gt[τ, i, 1]; δv_i = Vel_inc_gt[τ, i, 2];
	∂δui_∂ui = 1.0; ∂δvi_∂vi = 1.0;

	L_x = 0.0											# ∂L/∂x(τ)_i
	L_y = 0.0									  	 	# ∂L/∂y(τ)_i
	L_u = -∂klu_∂d(Vel_inc_pred, δu_i, τ) * ∂δui_∂ui		# ∂L/∂u(τ)_i
	L_v = -∂klv_∂d(Vel_inc_pred, δv_i, τ) * ∂δvi_∂vi		# ∂L/∂v(τ)_i
	return [L_x, L_y, L_u, L_v]
end


function ∂kl2_∂x_i(Vel_inc_gt, Vel_inc_pred, i)
	"""
	returns gradient of summand term in KL wrt x,y,u,v
	"""
	δu_i = Vel_inc_gt[T, i, 1]; δv_i = Vel_inc_gt[T, i, 2];
	∂δui_∂ui = 1.0; ∂δvi_∂vi = 1.0;

	δu_gt = Vel_inc_gt[T, :, 1]; δu_pr = Vel_inc_pred[T, :, 1];
	δv_gt = Vel_inc_gt[T, :, 2]; δv_pr = Vel_inc_pred[T, :, 2];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);

	function ∂klu_∂zpi(x, i)
		return -1/hu_pred * G_u(x) * (1/(N*hu_pred) * K_prime((δu_pr[i] - x)/hu_pred))/Gu_pred(x)
	end
	function ∂klv_∂zpi(x, i)
		return -1/hv_pred * G_v(x) * (1/(N*hv_pred) * K_prime((δv_pr[i] - x)/hv_pred))/Gv_pred(x)
	end

	L_x = 0.0											# ∂L/∂x(τ)_i
	L_y = 0.0									  	 	# ∂L/∂y(τ)_i
	L_u = ∂klu_∂zpi(δu_i, i) * ∂δui_∂ui		# ∂L/∂u(τ)_i
	L_v = ∂klv_∂zpi(δv_i, i) * ∂δvi_∂vi		# ∂L/∂v(τ)_i
	return [L_x, L_y, L_u, L_v]
end


function ∂kl2_∂x_i_t(Vel_inc_gt, Vel_inc_pred, i, τ)
	"""
	returns gradient of summand term in KL wrt x,y,u,v
	"""
	δu_i = Vel_inc_gt[τ, i, 1]; δv_i = Vel_inc_gt[τ, i, 2];
	∂δui_∂ui = 1.0; ∂δvi_∂vi = 1.0; #∂δwi_∂wi = 1.0;

	δu_gt = Vel_inc_gt[τ, :, 1]; δu_pr = Vel_inc_pred[τ, :, 1];
	δv_gt = Vel_inc_gt[τ, :, 2]; δv_pr = Vel_inc_pred[τ, :, 2];

	Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
	Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);

	function ∂klu_∂zpi(x, i)
		return -1/hu_pred * G_u(x) * (1/(N*hu_pred) * K_prime((δu_pr[i] - x)/hu_pred))/Gu_pred(x)
	end
	function ∂klv_∂zpi(x, i)
		return -1/hv_pred * G_v(x) * (1/(N*hv_pred) * K_prime((δv_pr[i] - x)/hv_pred))/Gv_pred(x)
	end

	L_x = 0.0											# ∂L/∂x(τ)_i
	L_y = 0.0									  	 	# ∂L/∂y(τ)_i
	L_u = ∂klu_∂zpi(δu_i, i) * ∂δui_∂ui		# ∂L/∂u(τ)_i
	L_v = ∂klv_∂zpi(δv_i, i) * ∂δvi_∂vi		# ∂L/∂v(τ)_i
	return [L_x, L_y, L_u, L_v]
end



function disc_compute_∇L(Vel_inc_gt, Vel_inc_pred, HT)
	∇L = zeros(n_params)
		for i ∈ 1 : N
			∂L_x = ∂kl2_∂x_i(Vel_inc_gt, Vel_inc_pred, i)
			for k in 1 : n_params
				∇L[k] += ∂L_x'*HT[T, i, :, k]
			end
		end
	if isnan(∇L[1]) || isinf(∇L[1])
		∇L = zeros(n_params);
	end
	return ∇L
end





function discKL_compute_∇L_t(Vel_inc_gt, Vel_inc_pred, HT)
	∇L = zeros(n_params)
	for τ in 1 : T
		for i ∈ 1 : N
			# ∂L_x = ∂kl2_∂x_i_t(Vel_inc_gt, Vel_inc_pred, i, τ)
			∂L_x = ∂kl_∂x_i_t(Vel_inc_gt, Vel_inc_pred, i, τ)
			for k in 1 : n_params
				∇L[k] += ∂L_x'*HT[τ, i, :, k]
			end
		end
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
	udif = V[i, 1] - V_gt[i, 1]
	vdif = V[i, 2] - V_gt[i, 2]

	L_x = (xdif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂x_i
	L_y = (ydif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂y_i
	L_u = (udif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂u_i
	L_v = (vdif)/(sqrt((xdif)^2 + (ydif)^2 + (zdif)^2 + (udif)^2 + (vdif)^2) + (wdif)^2) # ∂L/∂v_i
	return [L_x, L_y, L_u, L_v]
end


function ∂xmseᵢ(X, X_gt, V, V_gt, i)
	"""
	deriv of MSE
	"""
	xdif = X[i, 1] - X_gt[i, 1]
	ydif = X[i, 2] - X_gt[i, 2]
	udif = V[i, 1] - V_gt[i, 1]
	vdif = V[i, 2] - V_gt[i, 2]
	L_x = 2*(xdif)/(N*T) # ∂L/∂x_i
	L_y = 2*(ydif)/(N*T) # ∂L/∂y_i
	L_u = 2*(udif)/(N*T) # ∂L/∂u_i
	L_v = 2*(vdif)/(N*T) # ∂L/∂v_i
	return [L_x, L_y, L_u, L_v]
end



function compute_∂L_mse_k(traj_pred, traj_gt, vel_pred, vel_gt, HT)
	# global H_λ, H_U
	"""
	sums over t and i and compute the gradient wrt each param
	"""
	∇L = zeros(n_params)
	for t ∈ 1 : (T+1)
		for i ∈ 1 : N
			# L_x = ∂xLᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
			∂L_x = ∂xmseᵢ(traj_pred[t,:,:], traj_gt[t,:,:], vel_pred[t,:,:], vel_gt[t,:,:], i)
			for k ∈ 1 : n_params
				∇L[k] += ∂L_x' * HT[t, i, :, k]
			end
		end
	end
	return ∇L
end



function compute_∂L_∂θ_fixedτ(traj_pred, traj_gt, vel_pred, vel_gt, HT)
	# global H_λ, H_U
	"""
	sums over t and i and compute the gradient wrt each param
	"""
	∇L = zeros(n_params)
	for i ∈ 1 : N
		# L_x = ∂xLᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
		∂L_x = ∂xmseᵢ(traj_pred[τ,:,:], traj_gt[τ,:,:], vel_pred[τ,:,:], vel_gt[τ,:,:], i)
		for k ∈ 1 : n_params
			∇L[k] += ∂L_x' * HT[τ, i, :, k]
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
# gsp = 5; #produces grid of 2^gsp x 2^gsp x 2^gsp number of particles
# pgrid = Iterators.product((2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi)
# pgrid = vec(collect.(pgrid)) #particles
#
# X_grid = zeros(N, D);
# for n in 1 : N
#   X_grid[n, 1] = pgrid[n][1]
#   X_grid[n, 2] = pgrid[n][2]
# end

function obtain_grid(power)
  X_grid = zeros(N, D);
  for i in 1 : round(Int, 2^(power/2))
	  for j in 1 : round(Int, 2^(power/2))
	    n = round(Int, 2^(power/2)) * (i - 1) + j;
	    X_grid[n, 1] = 2. * pi * ((i - 1.) / sqrt(N));
	    X_grid[n, 2] = 2. * pi * ((j - 1.) / sqrt(N));
	  end
  end
  return X_grid
end


pow = floor(Int, log(N)/log(2))
X_grid = obtain_grid(pow)
display(Plots.scatter(X_grid[:, 1], X_grid[:, 2], ms=2.25));


n_hash = floor(Int, 2*pi / h);   l_hash = 2*pi / n_hash;
function obtain_interpolated_velocity(X_grid, X, V, rho)
  Vf = zeros(N,D);
  ∂Vf_∂x = zeros(N, D); ∂Vf_∂y = zeros(N, D);
  ∂Vf_∂u = zeros(N, D); ∂Vf_∂v = zeros(N, D);
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash];
  for n in 1 : N
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1], n);
  end
  XX = zeros(D);
  for n in 1 : N
    x_hash = [floor(Int, X_grid[n, 1] / l_hash) + 1,
              floor(Int, X_grid[n, 2] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
      xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
      while (xb_hash > n_hash)    xb_hash -= n_hash;   end
      for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
        yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
        while (yb_hash > n_hash)    yb_hash -= n_hash;   end
        for n2 in hash[xb_hash, yb_hash]
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
			end
			∂Vf_∂u[n, 1] = m * W(0.0, h) / rho[n2];
			∂Vf_∂v[n, 2] = m * W(0.0, h) / rho[n2];
          end
        end
    end   end
  end
  return Vf, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v
end


function obtain_interpolated_velocity_over_τ(traj, vels, rhos, T)
	Vf_t = zeros(T+1,N,D);
	∂Vft_∂x = zeros(T+1,N,D); ∂Vft_∂y = zeros(T+1,N,D);
	∂Vft_∂u = zeros(T+1,N,D); ∂Vft_∂v = zeros(T+1,N,D);
	for t in 1 : (T+1)
		Vf_t[t,:,:], ∂Vft_∂x[t,:,:], ∂Vft_∂y[t,:,:], ∂Vft_∂u[t,:,:], ∂Vft_∂v[t,:,:] =
		obtain_interpolated_velocity(X_grid, traj[t,:,:], vels[t,:,:], rhos[t,:])
	end
	return Vf_t, ∂Vft_∂x, ∂Vft_∂y, ∂Vft_∂u, ∂Vft_∂v
end




function ∂L_∂Y(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v, t, n)
	∂L_∂xi = 0.0; ∂L_∂yi = 0.0;
	∂L_∂ui = 0.0; ∂L_∂vi = 0.0;

	for i in 1 : D
		∂L_∂xi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂x[t, n, i])
		∂L_∂yi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂y[t, n, i])
		∂L_∂ui += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂u[t, n, i])
		∂L_∂vi += 2/N*(Vf_gt[t, n, i] - Vf_p[t, n, i]) * (-1*∂Vf_∂v[t, n, i])
	end
	return [∂L_∂xi, ∂L_∂yi, ∂L_∂ui, ∂L_∂vi]
end


function compute_∇L_field(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v, HT)
	∇L = zeros(n_params)
	for t in 1 : (T + 1)
		for n ∈ 1 : N
			∂L_x = ∂L_∂Y(Vf_gt, Vf_p, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v, t, n)
			for k ∈ 1 : n_params
				∇L[k] += ∂L_x' * HT[t, n, :, k]
			end
		end
	end
	return ∇L
end







#------Method dependendent loss functiong:

function compute_Lg(lg_method, Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	Lg = 0.0;
	if lg_method=="kl_t"
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T);
	end
	return Lg
end

function compute_L(loss_method, Vel_inc_gt, Vel_inc_pred, hu, hv, Gt_u, Gt_v, T)
	L = 0.0;
	if loss_method == "kl_lf"
		L = Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, hu, hv, Gt_u, Gt_v, T)
		# L_lf =
	end
	return L
end


function compute_L_og(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt, Gt_u, Gt_v, Gt_utest, Gt_vtest,
	Vel_inc_test, Vel_inc_model_test_th, Gu_test, Gv_test, hu_kde_test, hv_kde_test, traj_gt, traj_pred, T)
	L = 0.0; Lg = 0.0;
	if loss_method =="l2"
		L = mse(traj_gt[1:(T+1),:,:], traj_pred[1:(T+1), :,:]);
		# Lg = Ikl_fixed_τ(Vel_inc_test, Vel_inc_model_test_th, Gu_test, Gv_test, hu_kde_test, hv_kde_test, T)
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "kl"
		L = Ikl_fixed_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt, T)
		# Lg = Ikl_fixed_τ(Vel_inc_test, Vel_inc_model_test_th, Gu_test, Gv_test, hu_kde_test, hv_kde_test, T)
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "kl_t"
		L = Ikl_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt, Gt_u, Gt_v, T)
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "kl_t_one_dist"
		L = Ikl_τ_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt,  Gt_u, T)
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th,  hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "kl_one_dist"
		L = Ikl_one(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt, T)
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "kl_l2_t"
		Lkl = Ikl_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt, Gt_utest, Gt_vtest, T)
		Ll2 = mse(traj_gt[1:(T+1),:,:], traj_pred);
		L = Lkl + Ll2;
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "kl_lf_t"
		Vf_gt,d1,d2,d3,d4 = obtain_interpolated_velocity_over_τ(traj_gt, vels_gt, rhos_gt, T)
		Vfp, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		Lkl = Ikl_τ(Vel_inc_gt, Vel_inc_pred, G_u, G_v, hu_kde_gt, hv_kde_gt, T)
		Lf = mse(Vfp, Vf_gt);
		L = Lkl + Lf;
		Lg = Ikl_τ(Vel_inc_test, Vel_inc_model_test_th, hu_kde_test, hv_kde_test, Gt_utest, Gt_vtest, T)
	end
	if loss_method == "dkl_t"
		L = disc_kl_divergence_t(Vel_inc_gt, Vel_inc_pred)
	end
	if isinf(L) L = 1e2; end
	if isinf(Lg) Lg = 1e2; end
	return L, Lg
end

function compute_∇L(loss_method, Vel_inc_gt, Vel_inc_pred, traj_pred, traj_gt, vels_pred, vels_gt, HT)
	if loss_method == "l2"
		∇L = compute_∂L_mse_k(traj_pred, traj_gt, vels_pred, vels_gt, HT)
	end
	if loss_method == "kl"
		∇L = compute_∇L_KL(Vel_inc_gt, Vel_inc_pred, HT)
	end
	if loss_method == "kl_t"
		∇L = compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, HT)
	end
	if loss_method == "kl_t_one_dist"
		∇L = compute_∇L_KL_τ_one(Vel_inc_gt, Vel_inc_pred, HT)
	end
	if loss_method == "kl_one_dist"
		∇L = compute_∇L_KL_one(Vel_inc_gt, Vel_inc_pred, HT)
	end
	if loss_method == "kl_l2_t"
		∇L_kl = compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, HT)
		∇L_l2 = compute_∂L_mse_k(traj_pred, traj_gt, vels_pred, vels_gt, HT)
		∇L = ∇L_kl + ∇L_l2
	end
	if loss_method =="kl_lf_t"
		Vfp, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v =
		obtain_interpolated_velocity_over_τ(traj_pred, vels_pred, rhos_pred, T)
		Diff_pred, Vel_inc_pred = obtain_pred_dists(traj_pred, vels_pred, traj_gt[1,:,:], vels_gt[1,:,:])
		∇L_kl = compute_∇L_KL_τ(Vel_inc_gt, Vel_inc_pred, HT)
		∇L_lf = compute_∇L_field(Vf_gt, Vfp, ∂Vf_∂x, ∂Vf_∂y, ∂Vf_∂u, ∂Vf_∂v, HT)
		∇L = ∇L_kl + ∇L_lf
	end
	if loss_method == "dkl_t"
		∇L = discKL_compute_∇L_t(Vel_inc_gt, Vel_inc_pred, HT)
	end
	return ∇L
end




function rotational_metric_semi_inf(X, V, p, c, h, α, β, θ, model_A)
	R_90 = [0.0 -1.0; 1.0 0.0];
	Fry, rh_ = model_A((R_90*X')', (R_90*V')', p, c, h, α, β, θ)
    F, rh_ = model_A(X, V, p, c, h, α, β, θ)
    RF = (R_90 * F')'
	return mse(Fry, RF)
end

function rotational_metric_phys_inf(X, V, c, h, α, β, θ, model_A)
	R_90 = [0.0 -1.0; 1.0 0.0];
	Fry, rh_ = model_A((R_90*X')', (R_90*V')', α, β, h, c, g, θ)
    F, rh_ = model_A(X, V, α, β, h, c, g, θ)
    RF = (R_90 * F')'
	return mse(Fry, RF)
end


function rotational_metric(X, V, c, h, α, β, θ, model_A)
    Q,R = qr(randn(D,D)); Q = Q*Diagonal(sign.(diag(R))); #random orthogonal matrix
	R_90 = [0.0 -1.0; 1.0 0.0];
    Fqy, rh_ = model_A((Q*X')', (Q*V')', α, β, h, c, g, θ)
	Fry, rh_ = model_A((R_90*X')', (R_90*V')', α, β, h, c, g, θ)
    F, rh_ = model_A(X, V, α, β, h, c, g, θ)
    QF = (Q*F')'
    RF = (R_90 * F')'
	return mse(Fqy, QF), mse(Fry, RF)
end
