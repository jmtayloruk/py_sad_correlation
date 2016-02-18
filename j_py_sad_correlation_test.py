# Run the following command:
#	python setup.py build; python setup.py install
# in order to compile and install the j_py_sad_correlation module
from j_py_sad_correlation import *
import numpy as N
import time

a = N.random.randint(0, 100, (16,16))+0.0
b = N.random.randint(0, 100, (32,32))+0.0

start = time.time()
sad_using_c_code = sad_correlation(a, b)
print sad_using_c_code
print time.time() - start

ssd_using_c_code = ssd_correlation(a, b)
print ssd_using_c_code

start = time.time()
sad_using_python_code = N.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
for y in range(sad_using_python_code.shape[1]):
	for x in range(sad_using_python_code.shape[0]):
		sad_using_python_code[x,y] = sum(sum(abs(a - b[x:x+a.shape[0], y:y+a.shape[1]])))
print time.time() - start
print sad_using_python_code

print "success if these values are both zero:", (sad_using_python_code - sad_using_c_code).max(), (sad_using_python_code - sad_using_c_code).min()