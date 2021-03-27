import numpy as np
import matplotlib.pyplot as plt

activity_labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTRAIRS", "SITTING", "STANDING", "LAYING",
                   "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STANF"]


def plotting(info, n_exp, n_user, list_of_labels):
	# -------------- GET DATA FOR PLOTTING --------------
	t = np.arange(0, len(info) * 0.02, 0.02)
	axis = list(zip(*info))
	axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

	# -------------- PLOTTING -------------- a masterplan to take over the world :D
	fig, axs = plt.subplots(3)
	plt.figure()
	fig.suptitle("acc_exp" + str(n_exp) + "_user" + str(n_user) + ".txt")

	fig.set_figheight(5)
	fig.set_figwidth(25)

	axs[0].plot(t, axis_x, 'black')
	axs[1].plot(t, axis_y, 'black')
	axs[2].plot(t, axis_z, 'black')

	for i in range(len(list_of_labels)):
		axs[0].plot(t[list_of_labels[i][1]: list_of_labels[i][2]:], axis_x[list_of_labels[i][1]: list_of_labels[i][2]:])
		axs[1].plot(t[list_of_labels[i][1]: list_of_labels[i][2]:], axis_y[list_of_labels[i][1]: list_of_labels[i][2]:])
		axs[2].plot(t[list_of_labels[i][1]: list_of_labels[i][2]:], axis_z[list_of_labels[i][1]: list_of_labels[i][2]:])

		axs[0].text(t[list_of_labels[i][1]] * 0.01, .1, "Teste" + str(i), transform=axs[0].transAxes, rotation=45)

	axs[0].set_xlabel('Time (s)')
	axs[0].set_ylabel('ACC_X')

	axs[1].set_xlabel('Time (s)')
	axs[1].set_ylabel('ACC_Y')

	axs[2].set_xlabel('Time (s)')
	axs[2].set_ylabel('ACC_Z')

	plt.show()
