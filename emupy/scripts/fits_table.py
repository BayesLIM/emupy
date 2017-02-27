import astropy.io.fits as fits
import numpy as np


def fits_table(dictionary,keys,filename,clobber=False):
	''' Takes data as a dictionary and makes a fits table, keys is a list or array 
		with the column names in the order you want them to be added to the fits file
		data in dictionary should be a numpy.ndarray '''
	length = len(keys)
	cols = []
	for i in range(length):
		format = dictionary[keys[i]].dtype.name
		if 'int' in format:
			form = 'J'
		elif 'float' in format:
			form = 'D'
		elif 'string' in format:
			form = '10A'
		elif 'bool' in format:
			form = 'L'
		cols.append(fits.Column(name=keys[i],format=form,array=dictionary[keys[i]]))
	tbhdu = fits.BinTableHDU.from_columns(cols)
	tbhdu.writeto(filename,clobber=clobber)
	return

def fits_data(fits_data,elim_zeros=True):
	d = {}
	for i in range(len(fits_data.dtype)):
		if elim_zeros == True:
			try:
				d[fits_data.columns[i].name] = fits_data[fits_data.columns[i].name][np.where(fits_data[fits_data.columns[i].name]!=0)]
			except TypeError:
				d[fits_data.columns[i].name] = fits_data[fits_data.columns[i].name]
		else:
			d[fits_data.columns[i].name] = fits_data[fits_data.columns[i].name]
	return d

def fits_append(orig_table,new_dic,new_keys,filename,clobber=True):
	''' Takes an original fits record table and appends to it new columns stored in new_dic '''
	# First run fits record table into a dictionary with keys
	orig_keys = orig_table.columns.names
	orig_dic = dict(map(lambda x: (x,orig_table[x]), orig_keys))

	# First do original fits table
	length = len(orig_keys)
	cols = []
	for i in range(length):
		format = orig_dic[orig_keys[i]].dtype.name
		if 'int' in format:
			form = 'J'
		elif 'float' in format:
			form = 'D'
		elif 'string' in format:
			form = '10A'
		elif 'bool' in format:
			form = 'L'
		cols.append(fits.Column(name=orig_keys[i],format=form,array=orig_dic[orig_keys[i]]))

	# Now to new columns
	length = len(new_keys)
	for i in range(length):	
		format = new_dic[new_keys[i]].dtype.name
		if 'int' in format:
			form = 'J'
		elif 'float' in format:
			form = 'D'
		elif 'string' in format:
			form = '10A'
		elif 'bool' in format:
			form = 'L'
		cols.append(fits.Column(name=new_keys[i],format=form,array=new_dic[new_keys[i]]))

	tbhdu = fits.BinTableHDU.from_columns(cols)
	tbhdu.writeto(filename,clobber=clobber)

