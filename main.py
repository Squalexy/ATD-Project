import numpy as np
import functions

# %% 1 -- RETRIEVE DATA FROM TXT FILES

info_users = []
info_labels = np.genfromtxt("HAPT_DATA_set\\RawData\\labels.txt", dtype=int)
# print("Info:", info_labels)

n_exp = 26
n_user = 13

# ACCOUNT EXPERIENCE data
for i in range(8):
	info_users.append(
		np.genfromtxt("HAPT_data_set\\RawData\\acc_exp" + str(
			n_exp) + "_user" + str(n_user) + ".txt", dtype='float'))

	n_exp += 1
	if i % 2 == 1:
		n_user += 1

functions.main_menu()


# %% 2 -- PLOTTING

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

	functions.plotting(info_users[i], n_exp, n_user, list_of_labels)
	n_exp += 1
	if i % 2 == 1:
		n_user += 1
