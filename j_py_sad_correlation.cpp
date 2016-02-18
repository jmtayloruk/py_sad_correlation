#define PY_ARRAY_UNIQUE_SYMBOL j_sad_pyarray
#include "Common/JAssert.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "Common/JPythonCommon.h"

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
	
	// We expect a and b to be two-dimensional double arrays.
	// The following constructors will check those requirements
	JPythonArray2D<double> window1(a);
	JPythonArray2D<double> window2(b);
	if (PyErr_Occurred()) return NULL;
	
	if ((window1.NDims() != 2) || (window1.NDims() != 2))
	{
		PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Expected two 2D arrays as parameters");
		return NULL;
	}

	int maxDX = window2.Dims()[0] - window1.Dims()[0];
	int maxDY = window2.Dims()[1] - window1.Dims()[1];
	if ((maxDX < 0) || (maxDY < 0))
	{
		PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Expected second array to be bigger than or equal to first array");
		return NULL;
	}
	
	npy_intp output_dims[2] = { maxDX+1, maxDY+1 };
	PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_DOUBLE);
	JPythonArray2D<double> resultArray(result);

	// For every possible shift of 'a' relative to 'b', calculate the SAD
//	printf("Array sizes %ldx%ld, %ldx%ld\n", window1.Dims()[0], window1.Dims()[1], window2.Dims()[0], window2.Dims()[1]);
	int w1Width = window1.Dims()[0];
	int w1Height = window1.Dims()[1];
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
						sum += abs(window1[x][y] - window2[x+dx][y+dy]);
					}
			}
			else
			{
				// Sum of squared differences
				for (int y = 0; y < w1Height; y++)
					for (int x = 0; x < w1Width; x++)
					{
						double diff = (window1[x][y] - window2[x+dx][y+dy]);
						sum += diff*diff;
					}
			}
			resultArray[dx][dy] = sum;	
		}
	return PyArray_Return(result);
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
