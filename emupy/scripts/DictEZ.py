'''
This file contains functions that make it easier to work with creating Dictionaries on-the-fly
'''

## Import Modules ##
from collections import OrderedDict

def create(names,locals_vars):
	'''
	Creates a sub-set dictionary from 'locals_var', that has only variables defined in 'names'	
	names	: 1 dimensional list, with length N, dtype=str
	local_var : locals() instance from lowest namespace
	'''
	N = len(names)
	dictionary = OrderedDict()
	for i in range(N):
		within = names[i] in locals_vars
		if within == True:
			x = names[i]
			dictionary.update({x:locals_vars[x]})
	
	return dictionary
