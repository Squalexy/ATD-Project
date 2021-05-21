import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fftpack import fft, fftshift
from statistics import *

from scipy.signal import spectrogram
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import SelectKBest, f_classif

import tsfel
import pandas as pd

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
	fig.suptitle("acc_exp" + str(n_exp) + "_user" + str(n_user) + ".txt", weight='bold')

	fig.set_figheight(10)
	fig.set_figwidth(25)

	axs[0].plot(t, axis_x, 'black')
	axs[1].plot(t, axis_y, 'black')
	axs[2].plot(t, axis_z, 'black')

	for i in range(len(list_of_labels)):
		axs[0].plot(t[list_of_labels[i][1]: list_of_labels[i][2]:], axis_x[list_of_labels[i][1]: list_of_labels[i][2]:])
		axs[1].plot(t[list_of_labels[i][1]: list_of_labels[i][2]:], axis_y[list_of_labels[i][1]: list_of_labels[i][2]:])
		axs[2].plot(t[list_of_labels[i][1]: list_of_labels[i][2]:], axis_z[list_of_labels[i][1]: list_of_labels[i][2]:])

		axs[0].text(t[int((list_of_labels[i][2] + list_of_labels[i][1]) / 2)], + 0.5,
		            activity_labels[list_of_labels[i][0] - 1], transform=axs[0].transData, rotation=77.5,
		            fontsize="x-small")

		axs[1].text(t[int((list_of_labels[i][2] + list_of_labels[i][1]) / 2)], -0.5,
		            activity_labels[list_of_labels[i][0] - 1], transform=axs[1].transData, rotation=77.5,
		            fontsize="x-small")

		axs[2].text(t[int((list_of_labels[i][2] + list_of_labels[i][1]) / 2)], -0.5,
		            activity_labels[list_of_labels[i][0] - 1], transform=axs[2].transData, rotation=77.5,
		            fontsize="x-small")

	axs[0].set_ylabel('ACC_X')
	axs[1].set_ylabel('ACC_Y')
	axs[2].set_ylabel('ACC_Z')
	axs[2].set_xlabel('Time (s)', weight='bold')

	plt.show()


# %% Exercise 3

# Choose which window to apply the DFT
def get_window(option, size):
	if option.lower() == "rect":
		window_signal = signal.windows.boxcar(size)
	# window_title = plt.title("Rect window")
	elif option.lower() == "triang":
		window_signal = signal.windows.triang(size)
	# window_title = plt.title("Triang window")
	elif option.lower() == "gauss (size/5)":
		window_signal = signal.windows.gaussian(size, std=size / 5)
	elif option.lower() == "gauss (size/10)":
		window_signal = signal.windows.gaussian(size, std=size / 10)
	elif option.lower() == "gauss (size/2)":
		window_signal = signal.windows.gaussian(size, std=size / 2)
	# window_title = plt.title("Gauss window")
	elif option.lower() == "hamming":
		window_signal = signal.windows.hamming(size)
	# window_title = plt.title("Hamming window")
	else:
		print("Wrong input, window set to default value (hamming)")
		window_signal = signal.windows.gaussian(size, std=size / 5)
	# window_title = plt.title("Rect window")

	return window_signal


def calculate_dft(segment, axis_x, axis_y, axis_z, window_option):
	window = get_window(window_option, segment[4] - segment[3])

	axis_x_detrended = signal.detrend(axis_x[segment[3]:segment[4]], type="constant")
	axis_y_detrended = signal.detrend(axis_y[segment[3]:segment[4]], type="constant")
	axis_z_detrended = signal.detrend(axis_z[segment[3]:segment[4]], type="constant")

	y = axis_x_detrended
	axis_x_fft_w = abs(fftshift(fft(np.multiply(y, window))))

	y = axis_y_detrended
	axis_y_fft_w = abs(fftshift(fft(np.multiply(y, window))))

	y = axis_z_detrended
	axis_z_fft_w = abs(fftshift(fft(np.multiply(y, window))))

	return axis_x_fft_w, axis_y_fft_w, axis_z_fft_w


def plot_activity(experience, window):
	print(f"Plotting all activities of {experience[0][0]}_{experience[0][1]}...")
	for i in range(len(experience)):
		fig, axs = plt.subplots(3)
		plt.figure()
		fig.set_figheight(5)
		fig.set_figwidth(10)
		fig.suptitle("DFT - " + activity_labels[experience[i][2] - 1] + "(" + str(experience[i][0]) + "_" + str(
			experience[i][1]) + ")\nWINDOW: " + window)

		x = np.linspace(-25, 25, experience[i][4] - experience[i][3])

		axs[0].plot(x, experience[i][5], 'blue')
		axs[0].set_ylabel('axis_x')
		axs[0].set_xlim([0, 25])

		axs[1].plot(x, experience[i][6], 'orange')
		axs[1].set_ylabel('axis_y')
		axs[1].set_xlim([0, 25])

		axs[2].plot(x, experience[i][7], 'green')
		axs[2].set_xlabel('Frequency (Hz)')
		axs[2].set_ylabel('axis_z')
		axs[2].set_xlim([0, 25])

		plt.show()
	print("Plot successful!\n")


def plot_activity_nowindow(experience):
	for i in range(len(experience)):
		fig, axs = plt.subplots(3)
		plt.figure()
		fig.set_figheight(5)
		fig.set_figwidth(10)
		fig.suptitle("DFT - " + activity_labels[experience[i][2] - 1] + "(" + str(experience[i][0]) + "_" + str(
			experience[i][1]))

		x = np.linspace(-25, 25, experience[i][4] - experience[i][3])

		axs[0].plot(x, experience[i][5], 'blue')
		axs[0].set_ylabel('axis_x')
		axs[0].set_xlim([0, 25])

		axs[1].plot(x, experience[i][6], 'orange')
		axs[1].set_ylabel('axis_y')
		axs[1].set_xlim([0, 25])

		axs[2].plot(x, experience[i][7], 'green')
		axs[2].set_xlabel('Frequency (Hz)')
		axs[2].set_ylabel('axis_z')
		axs[2].set_xlim([0, 25])

		plt.show()
	print("Plot successful!\n")


# DFT for a single experience
def fourier_single(info_labels, label, window_option, info_user, n_exp, n_user):
	single_experience = []  # Lista para guardar os intervalos da atividade pretendida
	segment = []
	for lab in info_labels:
		if int(lab[0]) == n_exp and int(lab[1]) == n_user and int(lab[2]) - 1 == activity_labels.index(label):
			segment += [lab[0], lab[1], lab[2], lab[3], lab[4]]

			axis = list(zip(*info_user))
			axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

			fftx, ffty, fftz = calculate_dft(segment, axis_x, axis_y, axis_z, window_option)

			segment += [fftx, ffty, fftz]
		single_experience += [segment]

	return single_experience


# DFT for all experience
# Guarda array com informaçóes, de todas as experiências
# Cada experiência tem o seguinte array: [LABEL, XMIN, XMAX, DFTX, DFTY, DFTZ]
def fourier(info_labels, window, info_user):
	print("-------------------------------\nCalculating DFT for all experiencess...")

	all_experiences = []
	n_exp = 26
	n_user = 13

	for i in range(8):
		single_experience = []
		for lab in info_labels:
			segment = []  # Lista para guardar os intervalos da atividade pretendida
			if int(lab[0]) == n_exp and int(lab[1]) == n_user:
				segment += [lab[0], lab[1], lab[2], lab[3], lab[4]]
				axis = list(zip(*info_user[i]))
				axis_x, axis_y, axis_z = axis[0], axis[1], axis[2]

				fftx, ffty, fftz = calculate_dft(segment, axis_x, axis_y, axis_z, window)

				segment += [fftx, ffty, fftz]
				single_experience.append(segment)
		all_experiences.append(single_experience)

		# single_experience ---> [N_EXP, N_USER, LABEL, XMIN, XMAX, DFTX, DFTY, DFTZ]
		# all_experiences ---> [single_experience1, single_experience2, ...]

		n_exp += 1
		if i % 2 == 1:
			n_user += 1

	print("Operation successful!")
	print("SEGMENT: [n_exp, n_user, label, xmin, xmax, fftz, ffty, fftz]")
	print("Single experience: [segment1, segment2, semgent3,...]")
	print("All_experiences = [single_experience_1, single_experience_2,...]\n-------------------------------\n")
	return all_experiences


# Creates a biiig csv with features of info extracted from dataset
def extract_data(info_labels):
	info_labels = pd.DataFrame(info_labels, columns=['exp', 'user', 'activity', 'xmin', 'xmax'])

	features = []
	for i in range(info_labels.shape[0]):
		try:
			# Something here not working as it should. Experiences are getting numbers from 1 to 8, and
			# lines after have no info at all
			user = info_labels["user"][i]
			exp = info_labels["exp"][i]
			activity = info_labels["activity"][i]
			xmin = info_labels["xmin"][i]
			xmax = info_labels["xmax"][i]

			df = pd.read_csv('HAPT_data_set\\RawData\\acc_exp%02d_user%02d.txt' % (exp, user), sep=' ',
			                 names=['x', 'y', 'z'])

			# Retrieves a pre-defined feature configuration file to extract all available features
			cfg = tsfel.get_features_by_domain()

			# Extract features
			ts = tsfel.time_series_features_extractor(cfg, df.iloc[xmin:xmax, :], 50)
			# Uma cena destas para cada frequência para cada eixo
			# ts["minha dft"] = [fft_vals]
			features.append(ts)
		except:
			pass
	features = pd.concat(features).reset_index(drop=True)
	final = pd.concat([info_labels, features], axis=1, ignore_index=False)
	final.to_csv("features.csv", index=False)

	"""	
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)
	print(final)
	"""


# Para a tabela EXCEL:
# 3 maiores freqs, média e desvio padrão das freqs, tentar determinar empiricamente passos por minuto,

def feature_extraction(segment):
	# features -> [ [X[freq1, freq2, ... ], mean, stdev], [Y[freq1, freq2, ... ], mean, stdev], ...
	# ... [Z[freq1, freq2, ... ], mean, stdev  ]
	features = []

	for i in range(3):
		max_magnitude = max(segment[i + 5])
		freqs = []
		for j in range(len(segment[i + 5])):
			if segment[i + 5] > max_magnitude * 0.75:
				freqs.append(segment[i + 5])
		features.append([freqs])

	# get mean and stdev
	features[0].append(mean(segment[5]))
	features[0].append(pstdev(segment[5]))
	features[1].append(mean(segment[6]))
	features[1].append(pstdev(segment[6]))
	features[2].append(mean(segment[7]))
	features[2].append(pstdev(segment[7]))

	return features


def sklearn_feature_extraction(activity, t_size, debug=False):
	csv = pd.read_csv("features.csv")
	csv = csv.dropna(axis=1)  # apaga as linhas com valores em falta (Nan) ; axis = 1 é a coluna
	y = list(csv['activity'])

	# mudar valores de [] consoante o tipo de atividade que queremos analisar
	# y = [0 if x in [4, 5, 6] else 1 for x in y]  # atividades estáticas, 0 se for estática, 1 se n for
	if activity.upper() in activity_labels:
		y = [0 if x == (activity_labels.index(activity.upper()) + 1) else 1 for x in y]  # For single activity
	elif activity.lower() == "dynamic":
		y = [0 if x in [1, 2, 3] else 1 for x in y]  # atividades dinâmicas, 0 se for dinâmica, 1 se n for
	elif activity.lower() == "static":
		y = [0 if x in [4, 5, 6] else 1 for x in y]  # atividades estáticas, 0 se for estática, 1 se n for
	elif activity.lower() == "transition":
		y = [0 if x in [7, 8, 9, 10, 11, 12] else 1 for x in
		     y]  # atividades de transição, 0 se for transição, 1 se n for

	X = csv.drop('activity', axis=1)

	# ---------------- ALGORITMO SELECIONA AS MELHORES FEATURES ---------------- #
	selector = SelectKBest(f_classif, k=2)  # estimador da qualidade das features, escolhe as 3 melhores

	selector = selector.fit(X, y)
	# fit faz treino desse estimador, transform transforma matriz e apaga features que n interessam

	X_new = selector.transform(X)
	mask = selector.get_support()
	X = pd.DataFrame(X_new, columns=[f for b, f in zip(mask, X.columns) if b])

	if debug:
		print(X.columns)

	# ---------------- ALGORITMO SELECIONA AS FEATURES QUE QUISER ---------------- #
	"""
	X = X[['0_Fundamental Frequency', '0_Maximum Frequency', '0_Absolute Energy']]
	print(X)
	"""
	# ---------------------------------------------------------------------------- #

	# criar o modelo
	model = DecisionTreeClassifier()

	try:
		# pega em x e y ( 2 arrays ), parte. 90% vão para train, 10% para test
		# num faz a aproximacao, no outro verifica-se
		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

		# treinar o modelo com os pontos que tinha

		model.fit(x_train, y_train)

		if debug:
			# ver quao bom o modelo é
			print(model.score(x_test, y_test))

		plot_tree(model, feature_names=X.columns)
		if debug:
			plt.show()

		# compara as previsões com o que estaria certo e cria relatório sobre a qualidade do modelo
		y_pred = model.predict(x_test, y_test)
		if debug:
			print(classification_report(y_test, y_pred))

		tp = sum([1 if t == p == 0 else 0 for t, p in zip(y_test, y_pred)])
		fp = sum([1 if t != p == 0 else 0 for t, p in zip(y_test, y_pred)])
		tn = sum([1 if t == p == 1 else 0 for t, p in zip(y_test, y_pred)])
		fn = sum([1 if t != p == 1 else 0 for t, p in zip(y_test, y_pred)])

		sensitivity = tp / (tp + fn)
		specificity = tn / (tn + fp)

	except ZeroDivisionError:
		sensitivity = -1
		specificity = -1

	if debug:
		print(activity.upper() + ":\nSensivity:", sensitivity, "Specificity:", specificity, "\n")

	return sensitivity, specificity


def calculate_sensibility_specificity(activities):
	if activities.upper() == "ALL":
		activity_means = []
		for i in range(12):
			activity_means.append([[], []])
			for j in range(30):
				sensitivity, specificity = sklearn_feature_extraction(activity_labels[i], 0.3)
				if sensitivity != -1 and specificity != -1:
					activity_means[i][0].append(sensitivity)
					activity_means[i][1].append(specificity)
			activity_means[i][0] = mean(activity_means[i][0])
			activity_means[i][1] = mean(activity_means[i][1])

		for elem in activity_means:
			print(elem)

	else:
		activity_means = [[], []]
		for j in range(30):
			sensitivity, specificity = sklearn_feature_extraction(activities, 0.3)
			if sensitivity != -1 and specificity != -1:
				activity_means[0].append(sensitivity)
				activity_means[1].append(specificity)
		activity_means[0] = mean(activity_means[0])
		activity_means[1] = mean(activity_means[1])
		print(activity_means)


# Assume-se que as frequências com maior magnitude correspondem aos passos por segundo.
# Geralmente entre 1 e 2 HZs, percorremos as ffts das experiências do user, tiramos as frequência de maior
# magnitude, calculamos média
def get_max_frequencies(single_experience, debug=False):
	# para walking, walking upstairs e walking downstairs, respetivamente, cada com 3 eixos
	# max_freqs vai ter média das maiores frequências dos segmentos de cada atividade dinâmica
	max_freqs = [[[], [], []], [[], [], []], [[], [], []]]
	for segment in single_experience:
		# print("Segment: " + activity_labels[segment[2]-1])
		if activity_labels[segment[2] - 1] in dynamic_activity_labels:
			index = dynamic_activity_labels.index(activity_labels[segment[2] - 1])
			# print(activity_labels[segment[2]-1],np.where(segment[5] == max(segment[5]))[0][1] * 50 / (segment[4]-segment[3]-1) - 25)

			max_freqs[index][0].append(
				np.where(segment[5] == max(segment[5]))[0][1] * 50 / (segment[4] - segment[3] - 1) - 25)
			max_freqs[index][1].append(
				np.where(segment[6] == max(segment[6]))[0][1] * 50 / (segment[4] - segment[3] - 1) - 25)
			max_freqs[index][2].append(
				np.where(segment[7] == max(segment[7]))[0][1] * 50 / (segment[4] - segment[3] - 1) - 25)

	if debug:
		print("Frequencies before mean:", max_freqs)

	for i in range(len(max_freqs)):
		max_freqs[i][0] = mean(max_freqs[i][0])
		max_freqs[i][1] = mean(max_freqs[i][1])
		max_freqs[i][2] = mean(max_freqs[i][2])

	if debug:
		print("Frequencies after mean:", max_freqs)

	return max_freqs


def get_all_experience_steps(all_experiences, to_excel=False, debug=False):
	steps_by_experience = [get_max_frequencies(experience) for experience in all_experiences]
	if debug:
		for exp in steps_by_experience:
			print(exp)

	if to_excel:
		aux = []

		file = pd.DataFrame(steps_by_experience)

		"""for i in range(len(steps_by_experience)):
			aux.append([])
			for j in range(len(steps_by_experience[i])):
				for k in range(len(steps_by_experience[i][j])):
					aux[i].append(steps_by_experience[i][j][k])"""
		file = pd.concat([pd.DataFrame(file[i].to_list(), index=file.index, columns=["x", "y", "z"]) for i in range(3)],
		                 axis=1)

		tuples = [("WALKING", "X"), ("WALKING", "Y"), ("WALKING", "Z"), ("WALKING_UPSTAIRS", "X"),
		          ("WALKING_UPSTAIRS", "Y"),
		          ("WALKING_UPSTAIRS", "Z"), ("WALKING_DOWNSTAIRS", "X"), ("WALKING_DOWNSTAIRS", "Y"),
		          ("WALKING_DOWNSTAIRS", "Z")]

		file.columns = pd.MultiIndex.from_tuples(tuples)

		with pd.ExcelWriter("final.xlsx") as writer:
			file.to_excel(writer, sheet_name="Dynamic Activities Frequencies")

	return steps_by_experience


# %% Exercise 4

def calc_stft(info_user, n_exp):
	if n_exp % 2 == 0:
		n_user = n_exp - 13
	else:
		n_user = n_exp - 13 - 1

	axis = list(zip(*info_user))
	axis_z = np.array(axis[2])

	fs = 50
	Tframe = 1.5
	Toverlap = 0.1
	Nframe = int(np.round(Tframe * fs))
	Noverlap = int(np.round(Toverlap * fs))

	f, t, Sxx = spectrogram(axis_z, fs, window='boxcar', nperseg=Nframe,
	                        noverlap=Noverlap, detrend='constant', mode='magnitude')

	plt.figure(figsize=(30, 8))
	plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='plasma')
	plt.colorbar()
	plt.ylabel('f [Hz]', fontweight="bold")
	plt.xlabel('t [s]', fontweight="bold")
	plt.title("acc_exp" + str(n_exp) + "_user" + str(n_user) + "\nno window", fontweight="bold")
	plt.show()


# %% MENUS

def main_menu():
	print("Choose an option:\n"
	      "1. Plot all experiences\n"
	      "2. Re-calculate all experiences' DFT\n"
	      "3. Re-calculate single experience DFT\n"
	      "4. Plot single experience DFT\n"
	      "5. Get important features\n"
	      "6. Step calculation - frequency with biggest magnitude\n"
	      "7. Sensibility and specificity\n"
	      "9. Calculate STFT\n"
	      "10. Exit\n")


def single_dft_menu():
	print("input: <n_exp> <n_user> <label> <window>")


def all_dft_menu():
	print("input: <window>")


def experience_menu():
	print("input: <experience> (between 26 and 33)")


def feature_model_menu():
	print("1: Get features by single activity\n2: Get features by type\n")


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
