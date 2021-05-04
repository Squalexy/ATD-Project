import numpy as np
import matplotlib.pyplot as plt

activity_labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTRAIRS", "SITTING", "STANDING", "LAYING",
                   "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]


def plotting(info, n_exp, n_user, list_of_labels):
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


# Calculo da DFT para uma experiência
def fourier_single(label, window, step, overlap, info, n_exp, n_user):
	t = np.arange(0, len(info) * 0.02, 0.02)
	axis = list(zip(*info))
	axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]


# Chamar fourier_single para cada uma das várias experiências
def fourier(list_of_labels, window, step, overlap, info, n_exp, n_user):
	pass


def calc_dft(option=0):
	if option == 0:
		window = signal.windows.boxcar(51)
		plt.title("Rect window")
	elif option == 1:
		window = signal.windows.triang(51)
		plt.title("Triang window")
	elif option == 2:
		window = signal.windows.gaussian(51)
		plt.title("Gauss window")
	elif option == 3:
		window = signal.windows.hamming(51)
		plt.title("Hamming window")
	else:
		print("Wrong input, window set to default value (rect)")
		window = signal.windows.boxcar(51)
		plt.title("Rect window")

	# Precisamos mostrar isto??
	"""plt.plot(window)
	plt.ylabel("Amplitude")
	plt.xlabel("Sample")
	plt.figure()
	plt.show()
"""
# ------------------------------------------- MAIN MENU -------------------------------------------

def main_menu():
	print("Choose an option:\n"
	      "1. Plot all experiences\n"
	      "2. Calculate DFT\n"
	      "3. Calculate STFT\n"
	      "4. Exit")

# ------------------------------------------- DFT MENU -------------------------------------------

def dft_menu():
	print("Choose an option\n"
	      "1. DFT of a single experience\n"
	      "2. DFT of all experiences"
	      "3. Exit")

def single_dft_menu():
	print("Write 1 - n_exp   2 - n_User   3 - label   4 - \"Exit\"")


def plot_dft_menu():
	print("Write 1 - Window   2 - Step   3 - Overlap   4 - \"Exit\"")

def all_dft_menu():
