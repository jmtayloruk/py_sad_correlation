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
smallIWSize = 240
largeIWSize = 250
# Option to have the backing arrays larger, to test correct handling of strides
smallIWSize2 = smallIWSize + 10
largeIWSize2 = largeIWSize + 10


verbose = False
printPerformance = False
typesToUse = []
allTypes = ['uint8', 'uint16', 'int32']
for arg in sys.argv[1:]:
    if arg == '-v':
        verbose = True
        printPerformance = True
    elif arg == '-p':
        printPerformance = True
    elif arg == 'all':
        typesToUse += allTypes
    else:
        typesToUse.append(arg)

if (len(typesToUse) == 0):
    print("No data types specified - will test all (n.b. use optional -v to print intermediate results, or -p to print performance timings)")
    typesToUse += allTypes

def ReportError(typeToUse, description, result1, result2):
    success = np.all(result1 == result2)
    if success:
        print(" {0} {1} success".format(typeToUse, description))
        return 0
    else:
        print(" {0} {1} FAILURE".format(typeToUse, description))
        return 1

failureCount = 0
for typeToUse in typesToUse:
    print("Testing {0}:".format(typeToUse))

    ################# Generate test data #################
    # Generate two arrays containing random integers
    np.random.seed(1)
    if (typeToUse == 'uint8'):
        a2 = np.random.randint(0, 255, (smallIWSize2,smallIWSize2))
        b2 = np.random.randint(0, 255, (largeIWSize2,largeIWSize2))
    elif (typeToUse == 'uint16'):
        a2 = np.random.randint(0, 65535, (smallIWSize2,smallIWSize2))
        b2 = np.random.randint(0, 65535, (largeIWSize2,largeIWSize2))
    elif (typeToUse == 'int32'):
        # Yes. At present int32 is only specified to work with numbers up to 2^15-1
        a2 = np.random.randint(0, 65535, (smallIWSize2,smallIWSize2))
        b2 = np.random.randint(0, 65535, (largeIWSize2,largeIWSize2))
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
    start = time.perf_counter()
    sad_using_c_code = jps.sad_correlation(a, b)
    end = time.perf_counter()
    if verbose:
        print(' sad with c code: {0}'.format(sad_using_c_code))
    if printPerformance:
        print(' sad with c code took {0}'.format(end - start))

    ################# Call my module to compute the SSD #################
    start = time.perf_counter()
    ssd_using_c_code = jps.ssd_correlation(a, b)
    end = time.perf_counter()
    if verbose:
        print(' ssd with c code: {0}'.format(ssd_using_c_code))
        print(' ssd with c code took {0}'.format(end - start))


    ################# Compute SAD using pure python code, for comparison #################
    start = time.perf_counter()
    a = a.astype('int64')   # We need to convert to [signed] int64 to avoid encountering overflow issues in this python code
    b = b.astype('int64')
    sad_using_python_code = np.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
    for y in range(sad_using_python_code.shape[1]):
        for x in range(sad_using_python_code.shape[0]):
            sad_using_python_code[x,y] = np.sum(abs(a - b[x:x+a.shape[0], y:y+a.shape[1]]))
    end = time.perf_counter()
    if verbose:
        print(' sad with python code: {0}'.format(sad_using_python_code))
        print(' sad with python code took {0}'.format(end - start))
    failureCount += ReportError(typeToUse, "SAD", sad_using_c_code, sad_using_python_code)

    ################# Compute SSD using pure python code, for comparison #################
    start = time.perf_counter()
    a = a.astype('int64')   # We need to convert to [signed] int64 to avoid encountering overflow issues
    b = b.astype('int64')
    ssd_using_python_code = np.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
    for y in range(ssd_using_python_code.shape[1]):
        for x in range(ssd_using_python_code.shape[0]):
            ssd_using_python_code[x,y] = np.sum(((a - b[x:x+a.shape[0], y:y+a.shape[1]])**2))
    end = time.perf_counter()
    if verbose:
        print(' ssd with python code: {0}'.format(ssd_using_python_code))
        print(' ssd with python code took {0}'.format(end - start))
    failureCount += ReportError(typeToUse, "SSD", ssd_using_c_code, ssd_using_python_code)

    ################# Test a simpler calculation that this module can also perform #################
    # Although not actually PIV-specific, this module can also calculate the SAD between one image and a second array of multiple images
    # Test simpler SAD against reference frames
    np.random.seed(1)
    if (typeToUse != 'int32'):
        a2 = np.random.randint(0, 255, (smallIWSize,smallIWSize)).astype(typeToUse)
        b2 = np.random.randint(0, 255, (80,smallIWSize,smallIWSize)).astype(typeToUse)
        start = time.perf_counter()
        diffs_using_c_code = jps.sad_with_references(a2, b2);
        middle = time.perf_counter()
        diffs_using_python_code = np.sum(np.sum(np.abs(a2.astype('int')-b2.astype('int')), axis=2), axis=1)
        end = time.perf_counter()
        if verbose:
            print(' ref frame diffs with c code: {0}'.format(diffs_using_c_code))
            print(' ref frame diffs with python code: {0}'.format(diffs_using_python_code))
            print(' ref frame diffs with c code took: {0}'.format(middle-start))
            print(' ref frame diffs with python code: {0}'.format(end-middle))
        failureCount += ReportError(typeToUse, "Ref frame", diffs_using_c_code, diffs_using_python_code)
    elif verbose:
        print('Not testing diffs - that is only implemented for data type uint8')

    ################# Test error-catching #################
    for i in range(4):
        try:
            skipThisTest = False
            if i == 0:
                _ = jps.sad_correlation(np.zeros((10,10,3)).astype(typeToUse)[...,0], np.zeros((10,10)).astype(typeToUse))
            elif i == 1:
                _ = jps.sad_correlation(np.zeros((10,10)).astype(typeToUse), np.zeros((10,10,2)).astype(typeToUse)[...,0])
            elif (i == 2):
                if (typeToUse == 'int32'):
                    skipThisTest = True
                else:
                    _ = jps.sad_with_references(np.zeros((10,10,2)).astype(typeToUse)[...,0], np.zeros((80,10,10)).astype(typeToUse))
            elif (i == 3):
                if (typeToUse == 'int32'):
                    skipThisTest = True
                else:
                    _ = jps.sad_with_references(np.zeros((10,10)).astype(typeToUse), np.zeros((80,10,10,2)).astype(typeToUse)[...,0])
            if not skipThisTest:
                print(' FAILED to raise error on incorrect input')
                failureCount += 1
        except TypeError as e:
            print(' correctly raised error:', e)

if failureCount > 0:
    print("ERRORS OCCURRED DURING TESTING!")
else:
    print("All tests passed")
exit(failureCount)
