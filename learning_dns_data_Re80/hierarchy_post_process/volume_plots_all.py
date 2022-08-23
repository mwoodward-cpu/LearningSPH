import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
X, Y, Z = np.mgrid[0:2*np.pi:16j, 0:2*np.pi:16j, 0:2*np.pi:16j]
values = np.sin(X) * np.cos(Z) * np.sin(Y)
m_phys = ["phys_inf_W2ab_theta_po_liv_Pi", "phys_inf_Wab_theta_po_liv_Pi",
                  "phys_inf_Wliu_theta_po_liv_Pi", "phys_inf_theta_po_liv_Pi"]

m_nns = ["node_norm_theta_liv_Pi", "nnsum2_norm_theta_liv_Pi", "rot_inv_theta_liv_Pi",
                 "eos_nn_theta_alpha_beta_liv_Pi", "grad_p_theta_alpha_beta_liv_Pi"]
            
# method=m_phys[0]
# method=m_nns[4]
t = 1
method="dns"
f11 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")
# method=m_phys[1]
# f21 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")
# method=m_nns[4]
# f31 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")
# method=m_nns[5]
# f41 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")
# method=m_nns[3]
# f51 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")
# method=m_nns[2]
# f61 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")
# method=m_nns[1]
# f71 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")

t2 = 70
method="dns"
f12 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")
method=m_phys[0]
f22 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")
method=m_nns[4-1]
f32 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")
method=m_nns[5-1]
f42 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")
method=m_nns[3-1]
f52 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")
method=m_nns[2-1]
f62 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")
method=m_nns[1-1]
f72 = np.load(f"./field_data_snapshots/vf_u_t{t2}_" + method + ".npy")

t3 = 2*70
method="dns"
f13 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")
method=m_phys[1-1]
f23 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")
method=m_nns[4-1]
f33 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")
method=m_nns[5-1]
f43 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")
method=m_nns[3-1]
f53 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")
method=m_nns[2-1]
f63 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")
method=m_nns[1-1]
f73 = np.load(f"./field_data_snapshots/vf_u_t{t3}_" + method + ".npy")

t4 = 4*70
method="dns"
f14 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")
method=m_phys[1-1]
f24 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")
method=m_nns[4-1]
f34 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")
method=m_nns[5-1]
f44 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")
method=m_nns[3-1]
f54 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")
method=m_nns[2-1]
f64 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")
method=m_nns[1-1]
f74 = np.load(f"./field_data_snapshots/vf_u_t{t4}_" + method + ".npy")

t5 = 6*70
method="dns"
f15 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")
method=m_phys[1-1]
f25 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")
method=m_nns[4-1]
f35 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")
method=m_nns[5-1]
f45 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")
method=m_nns[3-1]
f55 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")
method=m_nns[2-1]
f65 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")
method=m_nns[1-1]
f75 = np.load(f"./field_data_snapshots/vf_u_t{t5}_" + method + ".npy")





def plot_volume_over_t(op, f1, f2, f3, f4, f5):
    win = 0.2
    fig = make_subplots(
    rows=1, cols=5,
    column_widths=[0.4, 0.4, 0.4, 0.4, 0.4],
    specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}]])#,
    # subplot_titles=("t = 0(s)", "t = 2.8(s)", "t = 5.6(s)", "t = 11.2(s)", "t = 16.8(s)"))

    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f1.flatten(),
        isomin=np.min(f1),
        isomax=np.max(f1),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f2.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=np.min(f2),
        isomax=np.max(f2),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
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
        # isomin=np.min(f3),
        # isomax=np.max(f3),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
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
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=1, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f5.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=1, col=5)

    h = 450
    fig.update_layout(
        # template="plotly_dark",
        autosize=False,
        width=5*h,
        height=h,
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=28,
            pad=26
        ),
        font=dict(
            size=14,
            color="Black"
        )
    )
    fig.update_layout(scene = dict(
                    xaxis_title=r'x',
                    yaxis_title=r'y',
                    zaxis_title=r'DNS: Truth'))

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
    fig.update_layout(scene5_xaxis_showticklabels=False,
                    scene5_yaxis_showticklabels=False,
                    scene5_zaxis_showticklabels=False)

    fig.update_annotations(font_size=28)
    fig.write_image(f"volume_figures/vf_u_" + method + ".png")

# plot_volume_over_t(0.2, vf_u1, vf_u2, vf_u3, vf_u4, vf_u5)




def plot_volume_over_all_t(op, f11, f12, f13, f14, f15, 
                                f21, f22, f23, f24, f25,
                                f31, f32, f33, f34, f35,
                                f41, f42, f43, f44, f45,
                                f51, f52, f53, f54, f55,
                                f61, f62, f63, f64, f65,
                                f71, f72, f73, f74, f75):
    win = 0.2
    fig = make_subplots(
    rows=7, cols=5,
    column_widths=[0.4, 0.4, 0.4, 0.4, 0.4],
    specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
            [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
            [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
            [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
            [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
            [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
            [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],])#,
    # subplot_titles=("t = 0(s)", "t = 2.8(s)", "t = 5.6(s)", "t = 11.2(s)", "t = 16.8(s)"))

    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f11.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f12.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f13.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f3),
        # isomax=np.max(f3),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=1, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f14.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=1, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f15.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=1, col=5)



#======================f2*====================================
    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f21.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=2, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f22.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=2, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f23.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=2, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f24.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=2, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f25.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=2, col=5)




#======================f3*====================================
    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f31.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=3, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f32.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=3, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f33.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=3, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f34.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=3, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f35.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=3, col=5)



#======================f4*====================================
    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f41.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=4, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f42.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=4, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f43.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=4, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f44.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=4, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f45.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=4, col=5)


#======================f5*====================================
    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f51.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=5, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f52.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=5, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f53.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=5, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f54.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=5, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f55.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=5, col=5)



#======================f6*====================================
    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f61.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=6, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f62.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=6, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f63.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=6, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f64.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=6, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f65.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=6, col=5)


#======================f7*====================================
    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f71.flatten(),
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=7, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f72.flatten(),
        # isomin=-win,
        # isomax=win,
        isomin=-0.18,
        isomax=0.18,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=7, col=2)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f73.flatten(),
        isomin=-win,
        isomax=win,
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        colorscale='jet',
        ), row=7, col=3)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f74.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f4),
        # isomax=np.max(f4),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        showscale=False,
        # colorscale='RdBu',
        colorscale='jet'
        ), row=7, col=4)

    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f75.flatten(),
        isomin=-win,
        isomax=win,
        # isomin=np.min(f5),
        # isomax=np.max(f5),
        opacity=op, # needs to be small to see through all surfaces
        surface_count=60, # needs to be a large number for good volume rendering
        colorscale='jet',
        ), row=7, col=5)


    h = 500
    fig.update_layout(
        # template="plotly_dark",
        autosize=False,
        width=5*h,
        height=7*h,
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=28,
            pad=26
        ),
        font=dict(
            size=14,
            color="Black"
        )
    )
    fig.update_layout(scene = dict(
                    xaxis_title=r'x',
                    yaxis_title=r'y',
                    zaxis_title=r'Method'))

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
    fig.update_layout(scene5_xaxis_showticklabels=False,
                    scene5_yaxis_showticklabels=False,
                    scene5_zaxis_showticklabels=False)

    fig.update_annotations(font_size=28)
    fig.write_image(f"volume_figures/vf_u_all.png")



plot_volume_over_all_t(0.2, f11, f12, f13, f14, f15, 
                                f11, f22, f23, f24, f25,
                                f11, f32, f33, f34, f35,
                                f11, f42, f43, f44, f45,
                                f11, f52, f53, f54, f55,
                                f11, f62, f63, f64, f65,
                                f11, f72, f73, f74, f75)














def plot_volume_field(field_data):
    fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=field_data.flatten(),
    # isomin=np.min(field_data),
    # isomax=np.max(field_data),
    isomin=-0.18,
    isomax=0.18,
    opacity=0.3, # needs to be small to see through all surfaces
    surface_count=60, # needs to be a large number for good volume rendering
    showscale=False,
    colorscale='RdBu'
    ))
    fig.update_layout(scene_xaxis_showticklabels=False,
                    scene_yaxis_showticklabels=False,
                    scene_zaxis_showticklabels=False)
    fig.update_layout(width=550, margin=dict(r=10, l=10, b=10, t=10))
    fig.write_image(f"volume_figures/vf_u_t{t}_" + method + ".png")
    # fig.show()

# plot_volume_field(vf_u1)



# def plot_volume_over_t(field1, field2):
#     fig = make_subplots(
#     rows=2, cols=2,
#     specs=[[{'type': 'volume'}, {'type': 'volume'}],
#            [{'type': 'volume'}, {'type': 'volume'}]])

#     fig.append_trace(go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=field1.flatten(),
#     isomin=-0.18,
#     isomax=0.18,
#     opacity=0.3, # needs to be small to see through all surfaces
#     surface_count=60, # needs to be a large number for good volume rendering
#     showscale=False,
#     colorscale='RdBu'
#     )), row=1, col=1)

#     fig.append_trace(go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=field2.flatten(),
#     isomin=-0.18,
#     isomax=0.18,
#     opacity=0.3, # needs to be small to see through all surfaces
#     surface_count=60, # needs to be a large number for good volume rendering
#     colorscale='RdBu'
#     )), row=1, col=2)

#     fig.append_trace(go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=field2.flatten(),
#     isomin=-0.18,
#     isomax=0.18,
#     opacity=0.3, # needs to be small to see through all surfaces
#     surface_count=60, # needs to be a large number for good volume rendering
#     colorscale='RdBu'
#     )), row=2, col=1)

#     fig.append_trace(go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=field2.flatten(),
#     isomin=-0.18,
#     isomax=0.18,
#     opacity=0.3, # needs to be small to see through all surfaces
#     surface_count=60, # needs to be a large number for good volume rendering
#     colorscale='RdBu'
#     )), row=2, col=2)

#     # fig.add_trace(go.Figure(data=go.Volume(
#     #     x=X.flatten(),
#     #     y=Y.flatten(),
#     #     z=Z.flatten(),
#     #     value=field1.flatten(),
#     #     isomin=-0.18,
#     #     isomax=0.18,
#     #     opacity=0.3, # needs to be small to see through all surfaces
#     #     surface_count=60, # needs to be a large number for good volume rendering
#     #     showscale=False,
#     #     colorscale='RdBu')),
#     #     row=1, col=1)

#     # fig.add_trace(go.Figure(data=go.Volume(
#     #     x=X.flatten(),
#     #     y=Y.flatten(),
#     #     z=Z.flatten(),
#     #     value=field2.flatten(),
#     #     isomin=-0.18,
#     #     isomax=0.18,
#     #     opacity=0.3, # needs to be small to see through all surfaces
#     #     surface_count=60, # needs to be a large number for good volume rendering
#     #     colorscale='RdBu')),
#     #     row=1, col=2)

#     fig.update_layout(height=600, width=1200, title_text="Side By Side Subplots")
#     fig.write_image(f"volume_figures/vf_u_" + method + ".png")

# #this does not work
# # plot_volume_over_t(vf_u1, vf_u2)




# def plot_iso_surface(field_data):
#     fig = go.Figure(data=go.Isosurface(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=field_data.flatten(),
#     isomin=-0.17,
#     isomax=0.17,
#     colorscale='RdBu',
#     caps=dict(x_show=False, y_show=False)
#     ))
#     fig.show()

# # plot_iso_surface(vf_u)


# # fig = go.Figure(data=go.Volume(
# #     x=X.flatten(),
# #     y=Y.flatten(),
# #     z=Z.flatten(),
# #     value=values.flatten(),
# #     isomin=-0.1,
# #     isomax=0.9,
# #     opacity=1.0, # needs to be small to see through all surfaces
# #     surface_count=30, # needs to be a large number for good volume rendering
# #     colorscale='RdBu'
# #     ))
# # fig.update_layout(scene_xaxis_showticklabels=False,
# #                   scene_yaxis_showticklabels=False,
# #                   scene_zaxis_showticklabels=False)
# # fig.show()
