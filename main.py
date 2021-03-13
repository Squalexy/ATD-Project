import numpy as np
import os.path
import matplotlib.pyplot as plt

# %% 1

info_users = []
n_exp = 26
n_user = 13

for i in range(8):
    info_users.append(
        np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + "\\HAPT_data_set\\RawData\\acc_exp" + str(
            n_exp) + "_user" + str(n_user) + ".txt",
                      dtype='float'))
    # print(f"{n_exp} -- {n_user}")
    n_exp += 1
    if i % 2 == 1:
        n_user += 1

# %% 2

t = np.arange(0, len(info_users[0]) * 0.02, 0.02)
axis = list(zip(*info_users[0]))
axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

fig, axs = plt.subplots(3)
plt.figure()

fig.set_figheight(5)
fig.set_figwidth(25)
axs[0].plot(t, axis_x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('ACC_X')

axs[1].plot(t, axis_y)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('ACC_Y')

axs[2].plot(t, axis_z)
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('ACC_Z')

plt.show()
