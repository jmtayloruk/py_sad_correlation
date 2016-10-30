#define PY_ARRAY_UNIQUE_SYMBOL j_sad_pyarray
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "common/jAssert.h"
#include "common/PIVImageWindow.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "common/jPythonCommon.h"

template<class TYPE> void SetImageWindowForPythonWindow(ImageWindow<TYPE> &imageWindow, JPythonArray2D<TYPE> &pythonWindow)
{
	imageWindow.baseAddr = pythonWindow.Data();
	imageWindow.width = pythonWindow.Dims()[1];
	imageWindow.height = pythonWindow.Dims()[0];
	imageWindow.elementsPerRow = pythonWindow.Strides()[0];
}

template<int correlationType, class TYPE> void correlation3(JPythonArray2D<TYPE> &window1, JPythonArray2D<TYPE> &window2, JPythonArray2D<double> &result)
{
	// This conversion between types is a bit silly, but I now want to convert my python arrays to ImageWindow objects,
	// since that's a generic type that I use in other code of mine as well.
	// (The overheads should be negligible compared to the actual calculation)
	ImageWindow<TYPE> imageWindow1;
	SetImageWindowForPythonWindow(imageWindow1, window1);
	ImageWindow<TYPE> imageWindow2;
	SetImageWindowForPythonWindow(imageWindow2, window2);
	ImageWindow<double> imageWindowResult;
	SetImageWindowForPythonWindow(imageWindowResult, result);
	CrossCorrelateImageWindows<correlationType, TYPE>(imageWindow1, imageWindow2, imageWindowResult);
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
	
	// Correlation code assumes the x dimension is contiguous
	// Note that although it now seems strange to deliberately do something different to how Python does it internally,
	// I have defined a stride of 1 (not sizeof(TYPE)) to represent contiguous array elements.
    if (window1.Strides()[1] != 1)
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Expected array 1 to be contiguous in x s[0]=%ld s[1]=%ld sizeof(type)=%zd itemsize=%ld", window1.Strides()[0], window1.Strides()[1], sizeof(TYPE), PyArray_ITEMSIZE(a));
        return NULL;
    }
    if (window2.Strides()[1] != 1)
    {
        PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Expected array 2 to be contiguous in x");
        return NULL;
    }
    
    npy_intp output_dims[2] = { maxDY+1, maxDX+1 };
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_DOUBLE);
    JPythonArray2D<double> resultArray(result);
    
    if (sad)
        correlation3<kCorrelationSAD>(window1, window2, resultArray);
    else
        correlation3<kCorrelationSSD>(window1, window2, resultArray);
    return PyArray_Return(result);
}

extern "C" PyObject *correlation(PyObject *self, PyObject *args, bool sad)
{
	// inputs
	PyArrayObject *a, *b;

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
