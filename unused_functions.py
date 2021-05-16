"""
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

	fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
	plt.show()
"""
