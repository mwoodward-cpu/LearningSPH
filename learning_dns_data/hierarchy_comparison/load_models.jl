using NPZ
using BSON: @load
using Flux

function load_phys_inf_learned_model(method, l_method, itr, lr, T, θ0, t_coarse, dt)
	pdir = "./learned_models_fin/phys"
	data_dir_phy = "$(pdir)/output_data_unif_tracers_forward_$(method)_$(l_method)_itr$(itr)_lr$(lr)_T$(T)_θ$(θ0)_h0.335_tcoarse$(t_coarse)_dt$(dt)_height6_m$(method)_l$(l_method)_klswitch0/"
	params_path = "$(data_dir_phy)/params_intermediate.npy"
	p_fin = npzread(params_path)
	return p_fin
end


function load_nn_learned_model(method, l_method, height, class, itr, lr, T, θ0, t_coarse, dt)
        pdir = "./learned_models_fin/$(class)"
	nn_data_dir = "$(pdir)/output_data_unif_tracers_forward_$(method)_$(l_method)_itr$(itr)_lr$(lr)_T$(T)_θ$(θ0)_h0.335_tcoarse$(t_coarse)_dt$(dt)_height$(height)_m$(method)_l$(l_method)_klswitch0"
	params_path = "$(nn_data_dir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	@load "$(nn_data_dir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
	return p_fin, NN, re
end

function obtain_height_itr(method)
	if method =="node_norm"
		# height = 6; itr = 1000; lr = 0.05; kl_s = 0;
		height = 6; itr = 800; lr = 0.001; kl_s = 0;
	elseif method=="eos_nn_theta"
		height = 8; itr = 800; lr = 0.05; kl_s = 1;
	elseif method=="grad_p_theta"
		height = 6; itr = 800; lr = 0.04; kl_s = 1;
	elseif method=="nnsum2_norm_theta"
		height = 6; itr = 400; lr = 0.01; kl_s = 0;
    else method=="rot_inv"
        height = 6; itr = 800; lr = 0.05; kl_s = 0;
	end
	return height, itr, lr, kl_s
end

function obtain_itr_lr(method)
	if method =="node_norm_theta_liv_Pi"
		itr = 500; lr = 0.002; θ0 = 0.0012; class = "node"; height = 6
	end
	if method =="nnsum2_norm_theta_liv_Pi"
		itr = 600; lr = 0.01; θ0 = 0.0012; class = "nnsum"; height = 6
	end
	if method =="rot_inv_theta_liv_Pi"
		itr = 500; lr = 0.02; θ0 = 0.0009; class = "rot"; height = 6
	end
	if method =="grad_p_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02; θ0 = 0.0009; class = "gradp"; height = 6
	end
	if method =="eos_nn_theta_alpha_beta_liv_Pi"
		itr = 500; lr = 0.02; θ0 = 0.0009; class = "eos"; height = 8
	end
	return itr, lr, θ0, class, height
end

# m_phys = ["phys_inf_theta", "phys_inf_theta_livescu_ext", "phys_inf_theta_correct_Pi", "phys_inf_Wab_theta", "phys_inf_W2ab_theta", "phys_inf_Wliu_theta"]
# m_phys_po = ["phys_inf_theta_po", "phys_inf_Wab_po_theta"];
# m_nn = ["node_norm", "node_norm_theta", "nnsum2_norm_theta", "grad_p_theta", "eos_nn_theta"]
#
# method = m_nn[1]
# height, itr = obtain_height_itr(method)
# p_fin, NN, re = load_nn_learned_model(method, height, itr)
