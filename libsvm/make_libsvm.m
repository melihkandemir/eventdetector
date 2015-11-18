function make_libsvm()
    architecture = computer('arch');
    switch(architecture)
        case 'win32'
            mex -O -c svm.cpp
            mex -O -c svm_model_matlab.c
            mex -O svmtrain.c svm.obj svm_model_matlab.obj
            mex -O svmpredict.c svm.obj svm_model_matlab.obj
            mex -O libsvmread.c
            mex -O libsvmwrite.c
        case 'win64'
            mex -O -c -largeArrayDims svm.cpp
            mex -O -c -largeArrayDims svm_model_matlab.c
            mex -O -largeArrayDims svmtrain.c svm.obj svm_model_matlab.obj
            mex -O -largeArrayDims svmpredict.c svm.obj svm_model_matlab.obj
            mex -O -largeArrayDims libsvmread.c
            mex -O -largeArrayDims libsvmwrite.c
        case 'glnx86'
            mex -O -c svm.cpp
            mex -O -c svm_model_matlab.c
            mex -O svmtrain.c svm.o svm_model_matlab.o
            mex -O svmpredict.c svm.o svm_model_matlab.o
            mex -O libsvmread.c
            mex -O libsvmwrite.c
        case 'glnxa64'
            mex -O -c -largeArrayDims svm.cpp
            mex -O -c -largeArrayDims svm_model_matlab.c
            mex -O -largeArrayDims svmtrain.c svm.o svm_model_matlab.o
            mex -O -largeArrayDims svmpredict.c svm.o svm_model_matlab.o
            mex -O -largeArrayDims libsvmread.c
            mex -O -largeArrayDims libsvmwrite.c
        case 'maci'
            mex -O -c svm.cpp
            mex -O -c svm_model_matlab.c
            mex -O svmtrain.c svm.o svm_model_matlab.o
            mex -O svmpredict.c svm.o svm_model_matlab.o
            mex -O libsvmread.c
            mex -O libsvmwrite.c
        case 'maci64'
            mex -O -c -largeArrayDims svm.cpp
            mex -O -c -largeArrayDims svm_model_matlab.c
            mex -O -largeArrayDims svmtrain.c svm.o svm_model_matlab.o
            mex -O -largeArrayDims svmpredict.c svm.o svm_model_matlab.o
            mex -O -largeArrayDims libsvmread.c
            mex -O -largeArrayDims libsvmwrite.c
    end
end