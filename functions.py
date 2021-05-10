import numpy as np
import matplotlib.pyplot as plt

activity_labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTRAIRS", "SITTING", "STANDING", "LAYING",
                   "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]

dynamic_activity_labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTRAIRS"]


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


def calculate_dft(interval, axis_x, axis_y, axis_z, window_option):
	window = get_window(window_option, interval[2] - interval[1])
	# Para o eixo axis_x  v

	x = np.linspace(-25, 25, interval[2] - interval[1], endpoint=False)

	y = axis_x[interval[1]: interval[2]]
	# axis_x_fft = abs(fftshift(fft(y)))

	axis_x_fft_w = abs(fftshift(fft(np.multiply(y, window))))

	y = axis_y[interval[1]: interval[2]]
	axis_y_fft_w = abs(fftshift(fft(np.multiply(y, window))))

	y = axis_z[interval[1]: interval[2]]
	axis_z_fft_w = abs(fftshift(fft(np.multiply(y, window))))

	return axis_x_fft_w, axis_y_fft_w, axis_z_fft_w


def plot_experience(single_experience):
	fig = plt.figure(figsize=(10, 5))
	gs = gridspec.GridSpec(3, len(single_experience))

	for i in range(3):
		ax = fig.add_subplot(gs[i, 0])
		x = np.linspace(-25, 25, single_experience[0][2] - single_experience[0][1])
		if i == 0:
			ax.set_ylabel("Axis_X")
			ax.plot(x, single_experience[0][3])
		elif i == 1:
			ax.set_ylabel("Axis_Y")
			ax.plot(x, single_experience[0][4])
		elif i == 2:
			ax.set_ylabel("Axis_Z")
			ax.set_xlabel(single_experience[0][0])
			ax.plot(x, single_experience[0][5])

		for j in range(1, len(single_experience)):
			ax = fig.add_subplot(gs[i, j])
			x = np.linspace(-25, 25, single_experience[j][2] - single_experience[j][1])
			if i == 0:
				ax.plot(x, single_experience[j][3])
			elif i == 1:
				ax.plot(x, single_experience[j][4])
			else:
				ax.plot(x, single_experience[j][5])
				ax.set_xlabel(single_experience[j][0])

	print("hey")
	fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
	print("oh")
	plt.show()
	print("let's go")


# DFT for a single experience
def fourier_single(info_labels, label, window_option, info_user, n_exp, n_user):
	label_intervals = []  # Lista para guardar os intervalos da atividade pretendida
	intervals = []
	for lab in info_labels:
		if int(lab[0]) == n_exp and int(lab[1]) == n_user and int(lab[2]) - 1 == activity_labels.index(label):
			intervals += [lab[2], lab[3], lab[4]]

			axis = list(zip(*info_user))
			axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

			fftx, ffty, fftz = calculate_dft(intervals, axis_x, axis_y, axis_z, window_option)

			intervals += [fftx, ffty, fftz]
		label_intervals += [intervals]

		axis_x_fft = abs(fftshift(fft(y)))
		axis_x_fft_w = abs(fftshift(fft(np.multiply(y, window))))

		plt.figure()
		plt.title("Activity " + label)
		plt.plot(x, axis_x_fft, 'b', x, axis_x_fft_w, 'r')
		plt.show()


# DFT for all experience
# Guarda array com informaçóes, de todas as experiências
# Cada experiência tem o seguinte array: [LABEL, XMIN, XMAX, DFTX, DFTY, DFTZ]
def fourier(info_labels, window, info_user):
	print("Calculating DFT for all experiencess...")

	all_experiences = []
	single_experience = []  # Lista para guardar todas as experiências
	n_exp = 26
	n_user = 13

	for i in range(8):
		for lab in info_labels:
			segment = []  # Lista para guardar os intervalos da atividade pretendida
			if int(lab[0]) == n_exp and int(lab[1]) == n_user:
				segment += [lab[2], lab[3], lab[4]]
				axis = list(zip(*info_user[i]))
				axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

				fftx, ffty, fftz = calculate_dft(segment, axis_x, axis_y, axis_z, window)

				segment += [fftx, ffty, fftz]
				single_experience.append(segment)
		all_experiences.append(single_experience)

		# single_experience ---> [LABEL, XMIN, XMAX, DFTX, DFTY, DFTZ]
		# all_experiences ---> [single_experience1, single_experience2, ...]

		n_exp += 1
		if i % 2 == 1:
			n_user += 1

	print("Operation successful!\nAll_intervals = [experience 1: [label, xmin, xmax, fftx, ffty, fftz], ...]")
	return all_experiences


# %% MENUS

def main_menu():
	print("Choose an option:\n"
	      "1. Plot all experiences\n"
	      "2. Re-calculate all experiences' DFT\n"
	      "3. Re-calculate single experience DFT\n"
	      "4. Plot single experience DFT\n"
	      "5. Exit")


def single_dft_menu():
	print("input: <n_exp> <n_user> <label> <window>")


def all_dft_menu():
	print("input: <window>")


def experience_menu():
	print("input: <experience> (between 26 and 33")


# %% Validate datas

def validate_data(n_exp, n_user, label):
	if 33 < n_exp < 26:
		print("Wrong n_exp, must be between 26 and 33. Try again...")
		return False
	if 16 < n_user < 13:
		print("Wrong n_user, must be between 13 and 16. Try again...")
		return False
	if label not in dynamic_activity_labels:
		print("Wrong label, must choose a valid dynamic activity. Try again...")
		return False
	return True


def validate_experience(n_exp):
	if 33 < n_exp < 26:
		print("Wrong n_exp, must be between 26 and 33. Try again...")
		return False
