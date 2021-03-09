#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include "omp.h"
#include "reader.hpp"
#include "pca.hpp"
using namespace Eigen;

/* PCAngsd C++ version
Depends on BLAS, LAPACK, OpenMP and Eigen.
*/
int main(int argc, char* argv[]) {
    std::cout << "PCAngsd v0.1 - Experimental C++ version.\n";

    // Initialize arguments and default values
    const char* beagle;
    int m, n, k, iter = 100, maf_iter = 200, power = 7;
    double maf = 0.05, tole = 1e-5, maf_tole = 1e-4;
    std::string out = "pcangsd";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-beagle") == 0 && argv[i+1]) {
            beagle = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && argv[i+1]) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && argv[i+1]) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-e") == 0 && argv[i+1]) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-iter") == 0 && argv[i+1]) {
            iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-maf_iter") == 0 && argv[i+1]) {
            maf_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-power") == 0 && argv[i+1]) {
            power = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-maf") == 0 && argv[i+1]) {
            maf = atof(argv[++i]);
        } else if (strcmp(argv[i], "-tole") == 0 && argv[i+1]) {
            tole = atof(argv[++i]);
        } else if (strcmp(argv[i], "-maf_tole") == 0 && argv[i+1]) {
            maf_tole = atof(argv[++i]);
        } else if (strcmp(argv[i], "-out") == 0 && argv[i+1]) {
            out = argv[++i];
        } else {
            std::cerr << "Unknown argument! " << argv[i] << "\n";
            exit(-1);
        }
    }

    // Define arrays
    double* L = new double[m*n*3]; // Keeping in 1D for contiguous memory
    double* f = new double[m];

    // Read data
    readBeagle(beagle, L, m, n);
    std::cout << "Parsed " << m << " sites, and " << n << " individuals.\n";

    // Estimate allele frequencies
    emFrequencies(L, f, m, n, maf_iter, maf_tole);

    // Filter based on MAF
    if (maf > 0.0) {
        m = filterArrays(L, f, m, n, maf);
        std::cout << m << " sites retained after filtering.\n";
    }

    // Initiate matrices for iterative PCA (SVD)
    MatrixXd E(m, n);
    MatrixXd P(m, n);
    MatrixXd C(n, n);

    // Estimate individual allele frequencies and GRM
    pcangsdAlgo(L, f, E, P, C, m, n, k, power, iter, tole);

    // Save GRM
    std::ofstream outmat(out + ".cov");
    if (outmat.is_open()) {
        outmat << C << "\n";
    }

    // Free arrays
    delete [] L;
    delete [] f;
    return 0;
}
