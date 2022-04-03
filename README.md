# PCAngsd
**Experimental C++ version**

Dependencies:
* BLAS (MKL, OpenBLAS, ...)
* LAPACK
* [Eigen](https://eigen.tuxfamily.org/) (Linear algebra library)

Easy install dependencies on a Linux system (OpenBLAS example):
```
sudo apt install libopenblas-dev liblapacke-dev
```

How to compile:
```
g++ pcangsd.cpp pca.cpp reader.cpp shared.cpp -o pcangsd -O3 -fopenmp -lz -lopenblas -llapacke
```

How to run (example):
```
./pcangsd -b input.gz -e 2 -o test
```
