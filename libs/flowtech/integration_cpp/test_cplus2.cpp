#include <Python.h>
#include <iostream>

using namespace std;

int main() {
    // Inicializa o interpretador Python
    Py_Initialize();

	 // Adiciona o diretório atual ao path do Python para que ele possa encontrar o arquivo my_functions.py
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");

    // Importa o módulo Python
    PyObject *pName = PyUnicode_FromString("test_python");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    // Chama a função add
    if (pModule != NULL) {
        PyObject *pFuncAdd = PyObject_GetAttrString(pModule, "test");
        if (pFuncAdd && PyCallable_Check(pFuncAdd)) {
            PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(30));
            PyObject *pValue = PyObject_CallObject(pFuncAdd, pArgs);
            Py_DECREF(pArgs);
            double result = PyFloat_AsDouble(pValue);
            if (PyErr_Occurred()) {
              PyErr_Print(); // Handle potential errors during conversion
            } else {
              // std::cout << "Result: " << result << std::endl;
            }
            Py_DECREF(pFuncAdd);
        }
        Py_DECREF(pModule);
    } else {
      PyErr_Print();
    }
    // Finaliza o interpretador Python
    Py_Finalize();

    return 0;
}
    
