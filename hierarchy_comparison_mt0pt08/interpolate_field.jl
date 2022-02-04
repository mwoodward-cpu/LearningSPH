
function obtain_uniform_mesh(D, gsp)
	# gsp produces grid of 2^gsp x 2^gsp x 2^gsp number of particles
	pgrid = Iterators.product((2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi, (2*pi/(2^gsp)):(2*pi/(2^gsp)):2*pi)
	pgrid = vec(collect.(pgrid)) #particles
	Nf = ceil(Int, (2^gsp)^3)
	X_grid = zeros(Nf, D);
	for n in 1 : Nf
	  X_grid[n, 1] = pgrid[n][3]
	  X_grid[n, 2] = pgrid[n][2]
	  X_grid[n, 3] = pgrid[n][1]
	end
	return X_grid
end

function W(r, h)
  sigma = 1/(pi*h^3)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

# H(r) = (d W / d r) / r
function H(r, h)
  sigma = 1/(pi*h^3)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * (-3. + 9. * q / 4.) / h^2);
end


n_hash = floor(Int, 2*pi / h);   l_hash = 2*pi / n_hash;
function obtain_interpolated_vel_field(X_grid_new, X, V, rho, Nf)
  Vf = zeros(Nf,D);
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
  for n in 1 : Nf
	# nk = ceil(Int, n/8);
    x_hash = [floor(Int, X_grid_new[n, 1] / l_hash),
  			floor(Int, X_grid_new[n, 2] / l_hash),
  			floor(Int, X_grid_new[n, 3] / l_hash)];
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
          close = true; r2 = 0.; n3 = ceil(Int, n2/8);
          for i in 1 : D
            XX[i] = X_grid_new[n, i] - X[n2, i];
            while (XX[i] > pi)   XX[i] -= 2. * pi;   end
            while (XX[i] < -pi)   XX[i] += 2. * pi;   end
            r2 += XX[i] * XX[i];
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)
			for i in 1 : D
	            Vf[n, i] += m * W(sqrt(r2), h) * V[n2, i] / rho[n2];
			end
          end
        end
    end   end   end
  end
  return Vf
end


function obtain_interpolated_vel_field_over_τ(X_grid_new, traj, vels, rhos, T, Nf)
	Vf_t = zeros(T+1,Nf,D);
	for t in 1 : (T+1)
		println("Time step =  ", t)
		Vf_t[t,:,:] = obtain_interpolated_vel_field(X_grid_new, traj[t,:,:], vels[t,:,:], rhos[t,:], Nf)
	end
	return Vf_t
end
