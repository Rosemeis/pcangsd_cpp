# PCAngsd
**Experimental C++ version**

Dependencies:
* BLAS (MKL, OpenBLAS, ...)
* LAPACK
* OpenMP (For parallelization)
* [Eigen](https://eigen.tuxfamily.org/) (Linear algebra library)

How to compile:
```
g++ pcangsd.cpp pca.cpp reader.cpp -o pcangsd -O3 -fopenmp -lz -lopenblas -llapacke
```

How to run (example):
```
./pcangsd -beagle input.gz -m 50000 -n 100 -e 2 -out test
```
