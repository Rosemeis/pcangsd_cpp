#include <iostream>
#include <string>
#include <armadillo>
#include <string.h>
#include "omp.h"
#include "reader.h"
#include "pca.h"

/* PCAngsd C++ version
Depends on BLAS, LAPACK, OpenMP and Armadillo.
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
    double* l = new double[m*n*3]; // Keeping in 1D for contiguous memory
    double* f = new double[m];

    // Read data
    readBeagle(beagle, l, m, n);
    std::cout << "Parsed " << m << " sites, and " << n << " individuals.\n";

    // Estimate allele frequencies
    emFrequencies(l, f, m, n, maf_iter, maf_tole);

    // Filter based on MAF
    if (maf > 0.0) {
        m = filterArrays(l, f, m, n, maf);
        std::cout << m << " sites retained after filtering.\n";
    }

    // Initiate matrices for iterative PCA (SVD)
    arma::mat e(m, n);
    arma::mat p(m, n);
    arma::mat u(m, k);
    arma::vec s(k);
    arma::mat v(n, k);
    arma::mat c(n, n);

    // Estimate individual allele frequencies and GRM
    pcangsdAlgo(l, f, e, p, u, s, v, c, m, n, k, power, iter, tole);

    // Save GRM
    c.save(out + ".cov", arma::raw_ascii);

    // Free arrays
    delete [] l;
    delete [] f;
    return 0;
}
