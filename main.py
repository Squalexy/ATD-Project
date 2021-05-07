import numpy as np
import functions

activity_labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTRAIRS", "SITTING", "STANDING", "LAYING",
                   "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]


def main():
	# retrieve data from txt files
	print("Obtaining data from txt files...")
	path_to_labels = "HAPT_DATA_set\\RawData\\labels.txt"
	path_to_exp = "HAPT_data_set\\RawData\\acc_exp"
	info_users, info_labels = functions.retrieve_data(path_to_labels, path_to_exp)
	print("Data retrieved!\n")

	# info_users -> acc_expXX_userYY.txt
	#                0.4333333437219827 0.01944444533917746 0.8930556332257938
	#                0.4277777761889662 0.01805555649491447 0.8763889306267443
	#                0.4402778031382534 -0.004166666912662869 0.9027777791608564
	#                0.4402778031382534 -0.004166666912662869 0.9027777791608564
	#                ...

	# info_labels -> labels.txt
	#                26 13 5 304 1423
	#                26 13 7 1574 1711
	#                26 13 4 1712 2616
	#                26 13 8 2617 2758
	#                ...

	while True:
		# main menu
		functions.main_menu()
		choice = int(input())

		if choice == 1:
			print("\nPlotting all experiences...")
			functions.ex2(info_labels, info_users)
			print("Plot successful!\n")

		elif choice == 2:
			functions.dft_menu()
			while True:

				choice = int(input())
				if choice == 1:

					functions.single_dft_menu()
					n_exp, n_user, label = int(input()), int(input()), str(input())
					while functions.validate_data(n_exp, n_user, label) is False:
						n_exp, n_user, label = int(input()), int(input()), str(input())

					functions.plot_dft_menu()
					window, step, overlap = str(input()), int(input()), int(input())

					functions.fourier_single(info_labels, label, window, step, overlap, info_users[n_exp - 26], n_exp, n_user)

				elif choice == 2:
					functions.plot_dft_menu()

				elif choice == 3:
					break
				elif choice == 4:
					exit(0)
				else:
					print("Wrong option. Try again...")

		elif choice == 3:
			pass

		elif choice == 4:
			break

		else:
			print("Wrong option. Try again...")


if __name__ == "__main__":
	main()
