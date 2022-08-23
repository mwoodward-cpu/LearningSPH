using NPZ
using BSON: @load
using Flux

function load_dirs_names_t20()
	dir_t20 = "learned_models_t20_minl"
	d1 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_W2ab_theta_po_liv_Pi_lf_skl0_itr2300_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_W2ab_theta_po_liv_Pi_llf_klswitch0"
	d2 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wab_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wab_theta_po_liv_Pi_llf_klswitch0"
	d3 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_Wliu_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mphys_inf_Wliu_theta_po_liv_Pi_llf_klswitch0"
	d4 = "$(dir_t20)/output_data_unif_tracers_forward_phys_inf_theta_po_liv_Pi_lf_skl0_itr3000_lr0.02_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mphys_inf_theta_po_liv_Pi_llf_klswitch0"
	d5 = "$(dir_t20)/output_data_unif_tracers_forward_node_norm_theta_liv_Pi_lf_itr800_lr0.002_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height8_mnode_norm_theta_liv_Pi_llf_klswitch0"
	d6 = "$(dir_t20)/output_data_unif_tracers_forward_nnsum2_norm_theta_liv_Pi_lf_itr600_lr0.01_T20_θ0.0012_h0.335_tcoarse1_dt0.04_height6_mnnsum2_norm_theta_liv_Pi_llf_klswitch0"
	d7 = "$(dir_t20)/output_data_unif_tracers_forward_rot_inv_theta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mrot_inv_theta_liv_Pi_llf_klswitch0"
	# d7 = "./learned_models_t50_accl/output_data_unif_tracers_forward_rot_inv_theta_liv_Pi_lf_skl1_itr1001_lr0.02_T50_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mrot_inv_theta_liv_Pi_llf_klswitch1"
	d8 = "$(dir_t20)/output_data_unif_tracers_forward_eos_nn_theta_alpha_beta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height8_meos_nn_theta_alpha_beta_liv_Pi_llf_klswitch0"
	d9 = "$(dir_t20)/output_data_unif_tracers_forward_grad_p_theta_alpha_beta_liv_Pi_lf_itr1000_lr0.02_T20_θ0.0009_h0.335_tcoarse1_dt0.04_height6_mgrad_p_theta_alpha_beta_liv_Pi_llf_klswitch0"
	return d1, d2, d3, d4, d5, d6, d7, d8, d9
end

# d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t20()

function load_learned_model_params(pdir)
	params_path = "$(pdir)/params_intermediate.npy"
	p_fin = npzread(params_path)
	return p_fin
end

function load_nn_learned_model(pdir)
	p_fin = load_learned_model_params(pdir)
	@load "$(pdir)/NN_model.bson" NN
	println(NN)
	p_, re = Flux.destructure(NN)   #flatten nn params
	n_params = size(p_fin)[1]
	return p_fin, NN, re
end
