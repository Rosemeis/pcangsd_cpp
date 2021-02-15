# PCAngsd
**Experimental C++ version**

Dependencies:
* BLAS (MKL, OpenBLAS, ...)
* LAPACK
* OpenMP (For parallelization)
* [Armadillo](http://arma.sourceforge.net/) (Linear algebra library)

How to compile:
```
g++ pcangsd.cpp pca.cpp reader.cpp -o pcangsd -O3 -fopenmp -lz -larmadillo
```

How to run (example):
```
./pcangsd -beagle input.gz -m 50000 -n 100 -e 2 -out test
```
