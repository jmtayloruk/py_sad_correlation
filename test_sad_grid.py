import j_py_sad_correlation as jps
import numpy as np
from numpy.lib.stride_tricks import as_strided
import sys, time, warnings

if (len(sys.argv) != 2):
    print("Run with one argument specifying the data type for the arrays")
    exit(1)
typeToUse = sys.argv[1]

# Generate two arrays containing random integers
if (typeToUse == 'uint8'):
    a2 = np.random.randint(0, 255, (250,250)).astype('uint8')
    b2 = np.random.randint(0, 255, (40,250,250)).astype('uint8')
elif (typeToUse == 'uint16'):
    a2 = np.random.randint(0, 65535, (250,250)).astype('uint16')
    b2 = np.random.randint(0, 65535, (40,250,250)).astype('uint16')
else:
    print("Valid types are uint8, uint16")
    exit(1);	# Nothing else currently supported.

# Do fast C-based SAD calculation on a 3D reference array
diffs_using_c_code = jps.sad_with_references(a2, b2);
#print(diffs_using_c_code)

# Do fast C-based SAD calculation on a list of 2D images
diffs_using_c_code_list = jps.sad_with_references(a2, list(b2))
#print(diffs_using_c_code_list)

# Do longhand Python calculation for comparison
diffs_using_python_code = np.sum(np.sum(np.abs(a2.astype('float')-b2.astype('float')), axis=2), axis=1)
#print(diffs_using_python_code)

if typeToUse == 'uint8':
    # Generate two arrays containing random integers
    if (typeToUse == 'uint8'):
        a2 = np.random.randint(0, 255, (50,250,250)).astype('uint8')
        b2 = np.random.randint(0, 255, (40,250,250)).astype('uint8')
    elif (typeToUse == 'uint16'):
        a2 = np.random.randint(0, 65535, (50,250,250)).astype('uint16')
        b2 = np.random.randint(0, 65535, (40,250,250)).astype('uint16')
    else:
        print("Valid types are uint8, uint16")
        exit(1);	# Nothing else currently supported.

    diffs_using_c_code2 = jps.sad_grid(a2, b2);
    #print(diffs_using_c_code2)
    diffs_using_python_code2 = np.zeros((len(b2),len(a2)),dtype='float')
    for i in range(len(a2)):
        for j in range(len(b2)):
            diffs_using_python_code2[j,i] = np.sum(np.abs(a2[i].astype('float')-b2[j].astype('float')))
    #print(diffs_using_python_code)

else:
    warnings.warn("Not testing sad_grid() - you need to specify a type of uint8 for that")
    diffs_using_python_code2 = np.zeros((1))
    diffs_using_c_code2 = np.zeros((1))
          

print ("success if these values are all zero:", (diffs_using_python_code - diffs_using_c_code).max(), (diffs_using_python_code - diffs_using_c_code).min())
print ("success if these values are all zero:", (diffs_using_python_code - diffs_using_c_code_list).max(), (diffs_using_python_code - diffs_using_c_code_list).min())
print ("success if these values are all zero:", (diffs_using_python_code2 - diffs_using_c_code2).max(), (diffs_using_python_code2 - diffs_using_c_code2).min())
