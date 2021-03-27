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

# LABELS data

# %% 2 -- PLOTTING

n_exp = 26
n_user = 13


for i in range(8):
	list_of_labels = []
	for lab in info_labels:
		if int(lab[0]) == n_exp and int(lab[1]) == n_user:
			list_of_labels += [[lab[2], lab[3], lab[4]]]
			# print(f"{lab[0]} {lab[1]}")
			# print(list_of_labels)

	functions.plotting(info_users[i], n_exp, n_user, list_of_labels)
	n_exp += 1
	if i % 2 == 1:
		n_user += 1
