import numpy as np
import matplotlib.pyplot as plt

activity_labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTRAIRS", "SITTING", "STANDING", "LAYING",
                   "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]


# %% Exercise 1

def retrieve_data(path_to_labels, path_to_exp):
	info_users = []
	info_labels = np.genfromtxt(path_to_labels, dtype=int)

	n_exp = 26
	n_user = 13

	# account experience data
	for i in range(8):
		info_users.append(np.genfromtxt(path_to_exp + str(n_exp) + "_user" + str(n_user) + ".txt", dtype='float'))
		n_exp += 1
		if i % 2 == 1:
			n_user += 1
	return info_users, info_labels


# %% Exercise 2

def ex2(info_labels, info_users):
	n_exp = 26
	n_user = 13

	for i in range(8):
		list_of_labels = []
		for lab in info_labels:

			# 26 13 5 304 1423 --> lab[0] = n_exp
			# 26 13 7 1574 1711 --> lab[1] = n_user
			# 26 13 4 1712 2616 --> lab[2] = label atividade
			# 26 13 8 2617 2758 --> lab[3] = xmin atividade
			# 26 13 5 2759 3728 --> lab[4] = xmax atividade

			if int(lab[0]) == n_exp and int(lab[1]) == n_user:
				list_of_labels += [[lab[2], lab[3], lab[4]]]

		# [[5, 304, 1423], [7, 1574, 1711],...]

		plotting_ex2(info_users[i], n_exp, n_user, list_of_labels)
		n_exp += 1
		if i % 2 == 1:
			n_user += 1


def plotting_ex2(info, n_exp, n_user, list_of_labels):
	# -------------- GET DATA FOR PLOTTING --------------
	t = np.arange(0, len(info) * 0.02, 0.02)
	axis = list(zip(*info))
	axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

	# -------------- PLOTTING --------------
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


# %% Exercise 3

# Choose which window to apply the DFT
def get_window(option):
	if option.lower() == "rect":
		window_signal = signal.windows.boxcar(51)
		window_title = plt.title("Rect window")
	elif option.lower() == "triang":
		window_signal = signal.windows.triang(51)
		window_title = plt.title("Triang window")
	elif option.lower() == "gauss":
		window_signal = signal.windows.gaussian(51)
		window_title = plt.title("Gauss window")
	elif option.lower() == "hamming":
		window_signal = signal.windows.hamming(51)
		window_title = plt.title("Hamming window")
	else:
		print("Wrong input, window set to default value (rect)")
		window_signal = signal.windows.boxcar(51)
		window_title = plt.title("Rect window")
	return window_signal, window_title


# DFT for a single experience
def fourier_single(label, window, step, overlap, info, n_exp, n_user):
	t = np.arange(0, len(info) * 0.02, 0.02)
	axis = list(zip(*info))
	axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]


# DFT for a single experience, applied to all experiences
def fourier(list_of_labels, window, step, overlap, info, n_exp, n_user):
	pass


# %% MENUS

def main_menu():
	print("Choose an option:\n"
	      "1. Plot all experiences\n"
	      "2. Calculate DFT\n"
	      "3. Calculate STFT\n"
	      "4. Exit")


def dft_menu():
	print("Choose an option [1-4]:\n"
	      "1. DFT of a single experience\n"
	      "2. DFT of all experiences"
	      "3. Go back"
	      "4. Exit")


def single_dft_menu():
	print("input: <n_exp> <n_user> <label>")


def plot_dft_menu():
	print("----------\nWindows available\n----------\n-> rect\n-> triang\n-> gauss\n-> hamming\n")
	print("input: <window> <step> <overlap>")


def all_dft_menu():
	pass


# %% Validate datas

def validate_data(n_exp, n_user, label):
	if 33 < n_exp < 26:
		print("Wrong n_exp, must be between 26 and 33. Try again...")
		return False
	if 16 < n_user < 13:
		print("Wrong n_user, must be between 13 and 16. Try again...")
		return False
	if label not in activity_labels:
		print("Wrong label, must choose a valid label. Try again...")
		return False
	return True
