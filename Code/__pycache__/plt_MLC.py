def testing_function():
	print("Hallo Welt")
	return

def plot_MLC_field(MLC_dat, JAWS_dat):

	MLC_dat = MLC_dat/10
	field = np.linspace(0,0.715*80,80)
	plt.bar(field,MLC_dat[1,:],color="w")
	plt.bar(field,15-MLC_dat[1,:],width=0.6,bottom=MLC_dat[1,:])
	plt.bar(field,MLC_dat[0,:],color="w")
	plt.bar(field,-15-MLC_dat[0,:],width=0.6,bottom=MLC_dat[0,:])

	ax = plt.gca()
	ax.add_patch(ptc.Rectangle((0,-15),57.2/2+JAWS_dat[0]/10.0,30,facecolor="g",alpha=0.7))
	ax.add_patch(ptc.Rectangle((57.2/2+JAWS_dat[1]/10.0,-15),57.2/2-JAWS_dat[1]/10.0,30,facecolor="g",alpha=0.7))

	plt.axis('equal')
	plt.tight_layout()
	plt.show()

	return