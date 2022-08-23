import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
X, Y, Z = np.mgrid[0:2*np.pi:16j, 0:2*np.pi:16j, 0:2*np.pi:16j]

#methods and file names
m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
                  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"]

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
                 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"]
            
def plot_volume_field_at_t(m_, f1, t):
    op = 0.2
    fig = make_subplots(
    rows=1, cols=1,
    column_widths=[1],
    specs=[[{'type': 'volume'}]])

    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f1.flatten(),
        isomin=-0.2,
        isomax=0.2,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=80, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=1)


    h = 400
    fig.update_layout(
        autosize=False,
        width=h,
        height=h,
        margin=dict(
            l=1,
            r=1,
            b=1,
            t=1,
            pad=1
        ),
        font=dict(
            size=1,
            color="Black"
        )
    )

    fig.update_layout(scene_xaxis_showticklabels=False,
                    scene_yaxis_showticklabels=False,
                    scene_zaxis_showticklabels=False)
    
    # camera = dict(eye=dict(x=6, y=5, z=4)) 
    camera = dict(eye=dict(x=1.32, y=1.32, z=1.1)) #looks best
    fig.update_layout(scene_camera=camera)

    fig.update_annotations(font_size=32)
    fig.write_image(f"volume_figures_seq/vf_u_" + m_ + "_" + t + ".png")



def plot_volume_field_t(field_data, method, t):
    fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=field_data.flatten(),
    isomin=-0.12,
    isomax=0.12,
    opacity=0.2, # needs to be small to see through all surfaces
    surface_count=100, # needs to be a large number for good volume rendering
    showscale=False,
    colorscale='jet'
    ))

    fig.update_layout(scene_xaxis_showticklabels=False,
                    scene_yaxis_showticklabels=False,
                    scene_zaxis_showticklabels=False)
    h=400  
    fig.update_layout(
        autosize=False,
        width=h,
        height=h,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
        font=dict(
            size=1,
            color="Black"
        )
    )

    camera = dict(eye=dict(x=1.32, y=1.32, z=1.1)) #looks best
    fig.update_layout(scene_camera=camera)
    fig.write_image(f"volume_figures_seq/avgx_V/"+str(t).zfill(3) + "avgx_V_" + method + ".png")




def obtain_plots_over_all_t(method_in):
    t_frame = np.arange(2, 500, 2)
    for t in t_frame:
        print("saving fig at t = ", t)
        t_in = str(t)
        f = np.load(f"./field_data_snapshots/avgx_V_t"+t_in+"_"+method_in+".npy")
        print("max f = ", np.max(f))
        plot_volume_field_t(f, method_in, t)

# obtain_plots_over_all_t("dns")









#======================================================
#======================================================
#======================================================
#======================================================
#======================================================
#======================================================
#======================================================




def plot_volume_over_mphys_t(t, op, f1, f2, f3, f4):
    win = 0.2
    sc = 90
    fig = make_subplots(
    rows=1, cols=4,
    column_widths=[0.25, 0.25, 0.25, 0.25],
    specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}]],
    # subplot_titles=("DNS", "SPH-W2", "EoS", "Grad P"))
    # subplot_titles=("DNS", "Rot-Inv", "NN Sum", "NODE"))
    subplot_titles=("DNS", "SPH-W2", "SPH-W1", "SPH-Wq"))

    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f1.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=sc, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f2.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=sc, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f3.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=sc, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f4.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=sc, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=1, col=4)

    h = 350
    fig.update_layout(
        # template="plotly_dark",
        autosize=False,
        width=4*h,
        height=h,
        margin=dict(
            l=1,
            r=1,
            b=1,
            t=34,
            # t = 34,
            pad=1
        ),
        font=dict(
            size=1,
            color="Black"
        )
    )

    fig.update_layout(scene_xaxis_showticklabels=False,
                    scene_yaxis_showticklabels=False,
                    scene_zaxis_showticklabels=False)
    fig.update_layout(scene2_xaxis_showticklabels=False,
                    scene2_yaxis_showticklabels=False,
                    scene2_zaxis_showticklabels=False)
    fig.update_layout(scene3_xaxis_showticklabels=False,
                    scene3_yaxis_showticklabels=False,
                    scene3_zaxis_showticklabels=False)
    fig.update_layout(scene4_xaxis_showticklabels=False,
                    scene4_yaxis_showticklabels=False,
                    scene4_zaxis_showticklabels=False)
    
    # camera = dict(eye=dict(x=5, y=4, z=3.5)) 
    camera = dict(eye=dict(x=1.32, y=1.22, z=1.1)) #looks best
    fig.update_layout(scene_camera=camera)
    fig.update_layout(scene2_camera=camera)
    fig.update_layout(scene3_camera=camera)
    fig.update_layout(scene4_camera=camera)

    fig.update_annotations(font_size=22)
    # fig.write_image(f"volume_figures_seq/all_m/phys/"+str(t).zfill(3) +"vf_u_physm_t.png")
    fig.write_image(f"volume_figures_seq/all_m/sph/"+str(t).zfill(3) +"vf_u_physm_t.png")



def obtain_plots_over_mphys_and_t():
    t_frame = np.arange(422, 500, 2)
    for t in t_frame:
        print("saving fig at t = ", t)
        t_in = str(t)
        f1 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_dns.npy")
        f2 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_phys_inf_W2ab_theta_po_liv_Pi.npy")
        f3 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_eos_nn_theta_alpha_beta_liv_Pi.npy")
        f4 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_grad_p_theta_alpha_beta_liv_Pi.npy")
        # f5 = np.load(f"./field_data_snapshots/avgx_V_t"+t_in+"_rot_inv_theta_liv_Pi.npy")
        # f6 = np.load(f"./field_data_snapshots/avgx_V_t"+t_in+"_node_norm_theta_liv_Pi.npy")
        plot_volume_over_mphys_t(t, 0.2, f1, f2, f3, f4)


# obtain_plots_over_mphys_and_t()


def obtain_plots_over_msph_and_t():
    t_frame = np.arange(2, 502, 2)
    for t in t_frame:
        print("saving fig at t = ", t)
        t_in = str(t)
        f1 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_dns.npy")
        f2 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_phys_inf_W2ab_theta_po_liv_Pi.npy")
        f3 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_phys_inf_Wab_theta_po_liv_Pi.npy")
        f4 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_phys_inf_Wliu_theta_po_liv_Pi.npy")
        # f5 = np.load(f"./field_data_snapshots/avgx_V_t"+t_in+"_rot_inv_theta_liv_Pi.npy")
        # f6 = np.load(f"./field_data_snapshots/avgx_V_t"+t_in+"_node_norm_theta_liv_Pi.npy")
        plot_volume_over_mphys_t(t, 0.2, f1, f2, f3, f4)

obtain_plots_over_msph_and_t()



def obtain_plots_over_mnns_and_t():
    t_frame = np.arange(446, 500, 2)
    for t in t_frame:
        print("saving fig at t = ", t)
        t_in = str(t)
        f1 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_dns.npy")
        # f2 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_phys_inf_W2ab_theta_po_liv_Pi.npy")
        # f3 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_eos_nn_theta_alpha_beta_liv_Pi.npy")
        # f4 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_grad_p_theta_alpha_beta_liv_Pi.npy")
        f2 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_rot_inv_theta_liv_Pi.npy")
        f3 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_nnsum2_norm_theta_liv_Pi.npy")
        f4 = np.load(f"./field_data_snapshots/vf_u_t"+t_in+"_node_norm_theta_liv_Pi.npy")
        plot_volume_over_mphys_t(t, 0.2, f1, f2, f3, f4)


# obtain_plots_over_mnns_and_t()