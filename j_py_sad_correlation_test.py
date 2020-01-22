################# Test correctness and performance of this module #################

# Run this python script with one parameter, specifying the type of the data to be processed.
# This can be one of:
#  'uint8', uint16', 'int32'


import j_py_sad_correlation as jps
import numpy as np
from numpy.lib.stride_tricks import as_strided
import time
import sys

# Size of the small and large IWs that will be tested with my code
smallIWSize = 32
largeIWSize = 64
# Option to have the backing arrays larger, to test correct handling of strides
smallIWSize2 = smallIWSize + 10
largeIWSize2 = largeIWSize + 10


if (len(sys.argv) != 2):
	print("Run with one argument specifying the data type for the arrays")
	exit(1)
typeToUse = sys.argv[1]

################# Generate test data #################
# Generate two arrays containing random integers
np.random.seed(1)
if (typeToUse == 'uint8'):
	a2 = np.round(np.random.randint(0, 255, (smallIWSize2,smallIWSize2))+0.0)
	b2 = np.round(np.random.randint(0, 255, (largeIWSize2,largeIWSize2))+0.0)
elif (typeToUse == 'uint16'):
	a2 = np.round(np.random.randint(0, 65535, (smallIWSize2,smallIWSize2))+0.0)
	b2 = np.round(np.random.randint(0, 65535, (largeIWSize2,largeIWSize2))+0.0)
elif (typeToUse == 'int32'):
	# Yes. At present int32 is only specified to work with numbers up to 2^16
	a2 = np.round(np.random.randint(0, 65535, (smallIWSize2,smallIWSize2))+0.0)
	b2 = np.round(np.random.randint(0, 65535, (largeIWSize2,largeIWSize2))+0.0)
else:
	print("Valid types are uint8, uint16, int32")
	exit(1);	# Nothing else currently supported.
a2 = a2.astype(typeToUse)
b2 = b2.astype(typeToUse)

# This code is not required for basic use of my module, but I have included it to test my code with arrays generated
# using this type of stride_tricks. I want to ensure that works because it's something I use within my own code that
# will call this module.
a = np.lib.stride_tricks.as_strided(a2, strides=(smallIWSize2*a2.itemsize, a2.itemsize), shape=(smallIWSize, smallIWSize) )
b = np.lib.stride_tricks.as_strided(b2, strides=(largeIWSize2*b2.itemsize, b2.itemsize), shape=(largeIWSize, largeIWSize) )

################# Call my module to compute the SAD #################
start = time.time()
sad_using_c_code = jps.sad_correlation(a, b)
print (time.time() - start)
print ('sad with c code:'.format(sad_using_c_code))

################# Call my module to compute the SSD #################
ssd_using_c_code = jps.ssd_correlation(a, b)
print ('ssd_using_c_code'.format(ssd_using_c_code))

################# Compute SAD using pure python code, for comparison #################
start = time.time()
a = a.astype('int')   # We need to convert to int to avoid encountering overflow issues
b = b.astype('int')
sad_using_python_code = np.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
for y in range(sad_using_python_code.shape[1]):
	for x in range(sad_using_python_code.shape[0]):
		sad_using_python_code[x,y] = sum(sum(abs(a - b[x:x+a.shape[0], y:y+a.shape[1]])))
print (time.time() - start)
print ('sad with python code: {0}'.format(sad_using_python_code))

################# Compute SSD using pure python code, for comparison #################
start = time.time()
a = a.astype('int')   # We need to convert to int to avoid encountering overflow issues
b = b.astype('int')
ssd_using_python_code = np.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
for y in range(ssd_using_python_code.shape[1]):
	for x in range(ssd_using_python_code.shape[0]):
		ssd_using_python_code[x,y] = sum(sum((a - b[x:x+a.shape[0], y:y+a.shape[1]])**2))
print (time.time() - start)
print ('ssd with python code: {0}'.format(ssd_using_python_code))

################# Test a simpler calculation that this module can also perform #################
# Although not actually PIV-specific, this module can also calculate the SAD between one image and a second array of multiple images
# Test simpler SAD against reference frames
np.random.seed(1)
if (typeToUse == 'uint8'):
	a2 = np.round(np.random.randint(0, 255, (smallIWSize,smallIWSize))+0.0)
	b2 = np.round(np.random.randint(0, 255, (20,smallIWSize,smallIWSize))+0.0)
	a2 = a2.astype(typeToUse)
	b2 = b2.astype(typeToUse)

	diffs_using_c_code = jps.sad_with_references(a2, b2);
	diffs_using_python_code = np.sum(np.sum(np.abs(a2.astype('int')-b2.astype('int')), axis=2), axis=1)
	print ("success if these values are all zero: {0} {1}".format((diffs_using_python_code - diffs_using_c_code).max(), (diffs_using_python_code - diffs_using_c_code).min()))
else:
	print ('Not testing diffs - you need to specify a type of uint8')

print ("success if these values are all zero: {0} {1}".format((sad_using_python_code - sad_using_c_code).max(), (sad_using_python_code - sad_using_c_code).min()))
print ("success if these values are all zero: {0} {1}".format((ssd_using_python_code - ssd_using_c_code).max(), (ssd_using_python_code - ssd_using_c_code).min()))


