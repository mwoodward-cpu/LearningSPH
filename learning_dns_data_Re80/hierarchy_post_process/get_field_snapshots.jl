using NPZ

l_method = "lf"; T_pred = 500; T_train = 20;
function load_vel_pr_field_data_save_at_t(method, t)
	nx = ny = nz = 16; D = 3;
	println("Obtaining snapshots for field:  ", method)
    data_dir_phy = "./learned_field_data"
    vfpr_path = "$(data_dir_phy)/Vf_pr_$(method)_$(l_method)_Tp$(T_pred)_Tt$(T_train).npy"
    vf_pr = npzread(vfpr_path)[t,:,:];
	Vf = reshape(vf_pr, (nx, ny, nz, D))
	Vf_u = Vf[:,:,:,1]; Vf_v = Vf[:,:,:,2]; Vf_w = Vf[:,:,:,3];

	npzwrite("./field_data_snapshots/vf_u_t$(t)_$(method).npy", Vf_u)
	npzwrite("./field_data_snapshots/vf_v_t$(t)_$(method).npy", Vf_v)
	npzwrite("./field_data_snapshots/vf_w_t$(t)_$(method).npy", Vf_w)
end

function load_vel_gt_field_data_save_at_t(t, method="dns")
	nx = ny = nz = 16; D = 3;
	println("Obtaining snapshots for field:  ", method)
    data_dir_phy = "./learned_field_data"
    vfgt_path = "$(data_dir_phy)/Vf_gt_Tp500.npy"
    vf_gt = npzread(vfgt_path)[t,:,:];
	Vf = reshape(vf_gt, (nx, ny, nz, D))
	Vf_u = Vf[:,:,:,1]; Vf_v = Vf[:,:,:,2]; Vf_w = Vf[:,:,:,3];

	npzwrite("./field_data_snapshots/vf_u_t$(t)_$(method).npy", Vf_u)
	npzwrite("./field_data_snapshots/vf_v_t$(t)_$(method).npy", Vf_v)
	npzwrite("./field_data_snapshots/vf_w_t$(t)_$(method).npy", Vf_w)
end

m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
		  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"];

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
		 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"];

m_tot = vcat(m_phys[1], m_nns);

load_vel_gt_field_data_save_at_t(20)

# for m_ in m_tot
# 	load_vel_pr_field_data_save_at_t(m_, 1)
# end

# for t in 70:70:500
# 	load_vel_gt_field_data_save_at_t(t)
# 	for m_ in m_tot
# 		load_vel_pr_field_data_save_at_t(m_, t)
# 	end
# end
