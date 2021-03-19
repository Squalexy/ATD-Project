import numpy as np
import functions

# %% 1 -- RETRIEVE DATA FROM TXT FILES

info_users = []
n_exp = 26
n_user = 13

for i in range(8):
    info_users.append(
        np.genfromtxt("HAPT_data_set\\RawData\\acc_exp" + str(
            n_exp) + "_user" + str(n_user) + ".txt",
                      dtype='float'))
    n_exp += 1
    if i % 2 == 1:
        n_user += 1

# %% 2 -- PLOTTING

n_exp = 26
n_user = 13
for i in range(8):
    functions.plotting(info_users[i], n_exp, n_user)
    n_exp += 1
    if i % 2 == 1:
        n_user += 1
