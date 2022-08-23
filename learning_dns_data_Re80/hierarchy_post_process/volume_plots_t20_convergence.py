import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
X, Y, Z = np.mgrid[0:2*np.pi:16j, 0:2*np.pi:16j, 0:2*np.pi:16j]

#methods and file names
m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
                  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"]

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
                 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"]
 








#======================================================
#======================================================
#======================================================
#======================================================
#======================================================
#======================================================
#======================================================




def plot_volume_over_mphys_t(t, op, f1, f2):
    win = 0.2
    sc = 90
    fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.5, 0.5],
    specs=[[{'type': 'volume'}, {'type': 'volume'}]],
    subplot_titles=("DNS", "SPH-W2"))

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

    h = 350
    fig.update_layout(
        # template="plotly_dark",
        autosize=False,
        width=2*h,
        height=h,
        margin=dict(
            l=1,
            r=1,
            b=1,
            t=34,
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

    
    # camera = dict(eye=dict(x=5, y=4, z=3.5)) 
    camera = dict(eye=dict(x=1.32, y=1.22, z=1.1)) #looks best
    fig.update_layout(scene_camera=camera)
    fig.update_layout(scene2_camera=camera)
    fig.update_annotations(font_size=22)
    fig.write_image(f"volume_figures_conv_70/"+str(t).zfill(3) +"vf_u_physm_conv.png")



def obtain_plots_over_mphys_and_t():
    idx_n = np.arange(20, 41, 1)
    for i in idx_n:
        print("saving fig at i = ", i)
        i_in = str(i)
        f1 = np.load(f"./vf_u_t70_dns.npy")
        f2 = np.load(f"./3d_conv_data/vf_u_t70_phys_inf_W2ab_theta_po_liv_Pi_itr"+i_in+".npy")
        plot_volume_over_mphys_t(i, 0.2, f1, f2)


obtain_plots_over_mphys_and_t()