#/bin/bash
/user/local/cuda8/bin/nvcc Octant.cu -o Octant_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 main.cpp Octant_g.cu.o -o tf_Octant_so.so -shared -fPIC -I /user/local/python3.6/site-packages/tensorflow/include -I /user/local/cuda8/include -I /user/local/python3.6//site-packages/tensorflow/include/external/nsync/public -lcudart -L /user/local/cuda8/lib64/ -L/user/local/python3.6//python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
