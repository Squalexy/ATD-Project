import numpy as np
import matplotlib.pyplot as plt


def plotting(info, n_exp, n_user):
    t = np.arange(0, len(info) * 0.02, 0.02)
    axis = list(zip(*info))
    axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

    fig, axs = plt.subplots(3)

    plt.figure()
    # font = {'color': 'white'}
    fig.suptitle("acc_exp" + str(n_exp) + "_user" + str(n_user) + ".txt")

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
