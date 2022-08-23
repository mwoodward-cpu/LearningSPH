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
method=m_nns[2]
# method="dns"

t = 1
f1 = np.load(f"./field_data_snapshots/vf_u_t{t}_" + method + ".npy")

t2 = 70
f2 = np.load(f"./field_data_snapshots_t50/vf_u_t{t2}_" + method + ".npy")

t3 = 2*70
f3 = np.load(f"./field_data_snapshots_t50/vf_u_t{t3}_" + method + ".npy")

t4 = 4*70
f4 = np.load(f"./field_data_snapshots_t50/vf_u_t{t4}_" + method + ".npy")

t5 = 6*70
f5 = np.load(f"./field_data_snapshots_t50/vf_u_t{t5}_" + method + ".npy")




def plot_volume_over_t(op, f1, f2, f3, f4, f5):
    win = 0.2
    fig = make_subplots(
    rows=1, cols=5,
    column_widths=[0.4, 0.4, 0.4, 0.4, 0.4],
    specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}]])
    # subplot_titles=("t = 0(s)", "t = 2.8(s)", "t = 8.4(s)", "t = 14.0(s)", "t = 19.6(s)"))

    fig.add_trace(go.Volume(    
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=f1.flatten(),
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
        value=f2.flatten(),
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

    h = 500
    fig.update_layout(
        # template="plotly_dark",
        autosize=False,
        width=5*h,
        height=h,
        margin=dict(
            l=20,
            r=20,
            b=15,
            t=15,
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
    fig.write_image(f"volume_figures_t50/vf_u_" + method + ".png")

plot_volume_over_t(0.2, f1, f2, f3, f4, f5)










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
