#define PY_ARRAY_UNIQUE_SYMBOL j_sad_pyarray
#include "common/jAssert.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "common/jPythonCommon.h"
//#include "emmintrin.h"
#include "tmmintrin.h"		// SSSE3 (supplemental SSE3)

inline long long ExtractLongLongPairSum(void *i)
{
	// _mm_extract_epi64 is SSE4.1 so we have to do this one by hand on most machines
	// TODO: if I have access to any machines that implement this intrinsic, I should implement the option of doing it properly...
	long long *l = (long long *)i;
	//	printf("%lld %lld  %llx %llx\n", l[0], l[1], l[0], l[1]);
	return l[0] + l[1];
}

inline int SumOver32BitInts(void *i)
{
    uint32_t *l = (uint32_t *)i;
    return l[0] + l[1] + l[2] + l[3];
}

inline int OrOver32BitInts(void *i)
{
    uint32_t *l = (uint32_t *)i;
    return l[0] | l[1] | l[2] | l[3];
}

// A bit of hoop-jumping here to obtain an abs function that can be called with both double and integer arguments
template<class TYPE> TYPE absFunc(TYPE a);
template<> double absFunc<double>(double a) { return fabs(a); }
template<> int absFunc<int>(int a) { return abs(a); }
// Things get messy for the 8-bit case because we were working with an unsigned type!
// Fortunately this shouldn't get called, since I have included a template specialization for that case.
template<> unsigned char absFunc<unsigned char>(unsigned char a) { ALWAYS_ASSERT(0); }

template<bool sad, class TYPE> void correlation3(JPythonArray2D<TYPE> &window1, JPythonArray2D<TYPE> &window2, JPythonArray2D<double> &result, int maxDX, int maxDY)
{
    // Generic version
    // For every possible shift of 'a' relative to 'b', calculate the SAD
    int w1Width = window1.Dims()[1];
    int w1Height = window1.Dims()[0];
    for (int dy = 0; dy <= maxDY; dy++)
        for (int dx = 0; dx <= maxDX; dx++)
        {
            double sum = 0;
            if (sad)
            {
                // Sum of absolute differences
                for (int y = 0; y < w1Height; y++)
                    for (int x = 0; x < w1Width; x++)
                    {
                        sum += absFunc<TYPE>(window1[y][x] - window2[y+dy][x+dx]);
                    }
            }
            else
            {
                // Sum of squared differences
                for (int y = 0; y < w1Height; y++)
                    for (int x = 0; x < w1Width; x++)
                    {
                        double diff = (window1[y][x] - window2[y+dy][x+dx]);
                        sum += diff*diff;
                    }
            }
            result[dy][dx] = sum;
        }
}

template<> void correlation3<true, unsigned char>(JPythonArray2D<unsigned char> &window1, JPythonArray2D<unsigned char> &window2, JPythonArray2D<double> &result, int maxDX, int maxDY)
{
    // Specialized version for SAD with 8-bit data
    // For every possible shift of 'a' relative to 'b', calculate the SAD
    int w1Width = window1.Dims()[1];
    int w1Height = window1.Dims()[0];
    for (int dy = 0; dy <= maxDY; dy++)
        for (int dx = 0; dx <= maxDX; dx++)
        {
            double sum = 0;
            __m128i sumVec = (__m128i)_mm_setzero_ps();
            for (int y = 0; y < w1Height; y++)
            {
                int x = 0;
                for (; x <= w1Width - 16; x += 16)
                    sumVec = _mm_add_epi64(sumVec, _mm_sad_epu8(_mm_loadu_si128((__m128i*)&window1[y][x]), _mm_loadu_si128((__m128i*)&window2[y+dy][x+dx])));
                for (; x < w1Width; x++)
                    sum += abs(window1[y][x] - window2[y+dy][x+dx]);
            }
            sum += ExtractLongLongPairSum(&sumVec);
            result[dy][dx] = sum;
        }
}

template<> void correlation3<true, unsigned short>(JPythonArray2D<unsigned short> &window1, JPythonArray2D<unsigned short> &window2, JPythonArray2D<double> &result, int maxDX, int maxDY)
{
    // Specialized version for SAD with 8-bit data
    // For every possible shift of 'a' relative to 'b', calculate the SAD
    int w1Width = window1.Dims()[1];
    int w1Height = window1.Dims()[0];
	__m128i ones = _mm_set1_epi16(1);

	if (maxDX * maxDY >= (1<<15))
		PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "WOAH - that's a seriously big correlation matrix! This integer-based SAD code only accepts IWs that lead to correlation matrices with up to 2^15 entries.");

	for (int dy = 0; dy <= maxDY; dy++)
        for (int dx = 0; dx <= maxDX; dx++)
        {
            double sum = 0;
            __m128i sumVec = (__m128i)_mm_setzero_ps();
            for (int y = 0; y < w1Height; y++)
            {
                int x = 0;
                for (; x <= w1Width - 8; x += 8)
				{
					__m128i a = _mm_loadu_si128((__m128i*)&window1[y][x]);
					__m128i b = _mm_loadu_si128((__m128i*)&window2[y+dy][x+dx]);
					__m128i sad = _mm_abs_epi16(_mm_sub_epi16(a, b));
					// Even though this seems a bit odd, it turns out that the madd instruction
					// accumulates its results in a 32-bit integer. Thus this actually achieves
					// exactly what we want, even though there is no dedicated instruction e.g.
					// to do 16-bit SAD.
					sumVec = _mm_add_epi16(_mm_madd_epi16(sad, ones), sumVec);
				}
				for (; x < w1Width; x++)
                    sum += abs(window1[y][x] - window2[y+dy][x+dx]);
            }
            sum += SumOver32BitInts(&sumVec);
            result[dy][dx] = sum;
        }
}

void Check16BitData(JPythonArray2D<int> &window1)
{
	// Although this is in principle unnecessary and therefore inefficient, I want to include a test to ensure no values
	// are larger than 2^16-1. The test should be quick, and it will catch what would otherwise be nasty bugs
	int w1Width = window1.Dims()[1];
	int w1Height = window1.Dims()[0];

	__m128i orVec = (__m128i)_mm_setzero_ps();
	int orRest = 0;
	for (int y = 0; y < w1Height; y++)
	{
		int x = 0;
		for (; x <= w1Width - 4; x += 4)
			orVec = _mm_or_si128(orVec, _mm_loadu_si128((__m128i*)&window1[y][x]));
		for (; x < w1Width; x++)
			orRest |= window1[y][x];
	}
	int result = orRest | OrOver32BitInts(&orVec);
	if (result & 0xFFFF0000)
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "ERROR - you passed in values greater than 2^16 - 1 to the fast SAD code!");
}

template<> void correlation3<true, int>(JPythonArray2D<int> &window1, JPythonArray2D<int> &window2, JPythonArray2D<double> &result, int maxDX, int maxDY)
{
    // Specialized version for SAD with 32-bit data, BUT we assume we will not overflow an int when we sum across a small IW.
    // This probably implies that it should be used with 16-bit input data, and small IW pixel counts <=2^16 !
    
    // For every possible shift of 'a' relative to 'b', calculate the SAD
    int w1Width = window1.Dims()[1];
    int w1Height = window1.Dims()[0];
	
	Check16BitData(window1);
	Check16BitData(window2);
	if (maxDX * maxDY >= (1<<15))
		PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "WOAH - that's a seriously big correlation matrix! This integer-based SAD code only accepts IWs that lead to correlation matrices with up to 2^15 entries.");

	// Now get down to business!
	for (int dy = 0; dy <= maxDY; dy++)
        for (int dx = 0; dx <= maxDX; dx++)
        {
            double sum = 0;
            __m128i sumVec = (__m128i)_mm_setzero_ps();
            for (int y = 0; y < w1Height; y++)
            {
                int x = 0;
                for (; x <= w1Width - 4; x += 4)
                    sumVec = _mm_add_epi32(sumVec, _mm_abs_epi32(_mm_sub_epi32(_mm_loadu_si128((__m128i*)&window1[y][x]), _mm_loadu_si128((__m128i*)&window2[y+dy][x+dx]))));
                for (; x < w1Width; x++)
                    sum += abs(window1[y][x] - window2[y+dy][x+dx]);
            }
            sum += SumOver32BitInts(&sumVec);
            result[dy][dx] = sum;
        }
}

template<class TYPE> PyObject *correlation2(PyArrayObject *a, PyArrayObject *b, bool sad)
{
    // We expect a and b to be two-dimensional double arrays.
    // The following constructors will check those requirements
    JPythonArray2D<TYPE> window1(a);
    JPythonArray2D<TYPE> window2(b);
    if (PyErr_Occurred()) return NULL;
    
    if ((window1.NDims() != 2) || (window1.NDims() != 2))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Expected two 2D arrays as parameters");
        return NULL;
    }
    
    int maxDX = window2.Dims()[1] - window1.Dims()[1];
    int maxDY = window2.Dims()[0] - window1.Dims()[0];
    if ((maxDX < 0) || (maxDY < 0))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Expected second array to be bigger than or equal to first array");
        return NULL;
    }
	
    if ((PyArray_ITEMSIZE(a) != sizeof(TYPE)) || (PyArray_ITEMSIZE(b) != sizeof(TYPE)))
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Something weird happened with item sizes %d and %d, relative to expected size %d", (int)PyArray_ITEMSIZE(a), (int)PyArray_ITEMSIZE(b), (int)sizeof(TYPE));
        return NULL;
    }
    
    npy_intp output_dims[2] = { maxDY+1, maxDX+1 };
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_DOUBLE);
    JPythonArray2D<double> resultArray(result);
    
    if (sad)
        correlation3<true>(window1, window2, resultArray, maxDX, maxDY);
    else
        correlation3<false>(window1, window2, resultArray, maxDX, maxDY);
    return PyArray_Return(result);
}

extern "C" PyObject *correlation(PyObject *self, PyObject *args, bool sad)
{
	// inputs
	PyArrayObject *a, *b;		// list of coordinates for the input polygons (2d double numpy array)

	// parse the input arrays from *args
	if (!PyArg_ParseTuple(args, "O!O!", 
			&PyArray_Type, &a, 
			&PyArray_Type, &b))
	{
		PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unable to parse array!");
		return NULL;
	}
	
    if (PyArray_TYPE(a) == ArrayType<double>())
        return correlation2<double>(a, b, sad);
    else if (PyArray_TYPE(a) == ArrayType<unsigned char>())
        return correlation2<unsigned char>(a, b, sad);
    else if (PyArray_TYPE(a) == ArrayType<unsigned short>())
        return correlation2<unsigned short>(a, b, sad);
    else if (PyArray_TYPE(a) == ArrayType<int>())
        return correlation2<int>(a, b, sad);
    else
    {
		printf("Strides: %d %d\n", (int)PyArray_STRIDES(a)[0], (int)PyArray_STRIDES(a)[1]);
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Unsuitable array type %d passed in", PyArray_TYPE(a));
        return NULL;
    }
}

extern "C" PyObject *sad_correlation(PyObject *self, PyObject *args)
{
	return correlation(self, args, true);
}

extern "C" PyObject *ssd_correlation(PyObject *self, PyObject *args)
{
	return correlation(self, args, false);
}


/* Define a methods table for the module */

static PyMethodDef corr_methods[] = {
	{"sad_correlation", sad_correlation, METH_VARARGS},	
	{"ssd_correlation", ssd_correlation, METH_VARARGS},	
	{NULL,NULL} };



/* initialisation - register the methods with the Python interpreter */

extern "C" void initj_py_sad_correlation(void)
{
	(void) Py_InitModule("j_py_sad_correlation", corr_methods);
	import_array();
}
