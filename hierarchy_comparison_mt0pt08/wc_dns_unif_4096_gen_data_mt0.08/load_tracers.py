import numpy as np

u=np.fromfile('./vel_traj_filter4_4096_mesh_X0074.bin')
x=np.fromfile('./pos_traj_filter4_4096_mesh_X0074.bin')
r=np.fromfile('./rho_traj_filter4_4096_mesh_X0074.bin')
p=np.fromfile('./p_traj_filter4_4096_mesh_X0074.bin')

print(u.shape)
print("t_guess = ", u.size/(4096*3))

n_time = 500
N = 4096
D = 3
u=u.reshape([n_time,N,D])
x=x.reshape([n_time,N,D])
r=r.reshape([n_time,N])
p=p.reshape([n_time,N])

np.save('./vel_traj_4k.npy', u)
np.save('./pos_traj_4k.npy', x)
np.save('./rho_traj_4k.npy', r)
np.save('./p_traj_4k.npy', p)


