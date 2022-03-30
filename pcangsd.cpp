#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <string.h>
#include <zlib.h>
#include "omp.h"
#include "reader.hpp"
#include "pca.hpp"
#include "shared.hpp"
using namespace Eigen;

// Info function
void info() {
	std::cerr << "Arguments:\n";
	std::cerr << "\t--beagle (-b), Path to genotype likelihood file (*.beagle.gz)\n";
	std::cerr << "\t--n_eig (-e), Number of eigenvectors to use (default: 2)\n";
	std::cerr << "\t--out (-o), File-prefix for output files (default: pcangsd)\n";
	std::cerr << "\t--threads (-t), Number of threads (default: 1)\n";
	std::cerr << "Parameters:\n";
	std::cerr << "\t--hwe, Estimate per-site inbreeding coefficients\n";
	std::cerr << "\t--selection, Perform genome-wide selection scan\n";
	std::cerr << "\t--loadings, Output SNP loadings of inner SVD\n";
	std::cerr << "\t--iter, Maximum number of iterations for iterative SVD (default: 100)\n";
	std::cerr << "\t--maf_iter, Maximum number of EM iterations for allele frequencies (default: 200)\n";
	std::cerr << "\t--hwe_iter, Maximum number of EM iterations for inbreeding coefficients (default: 200)\n";
	std::cerr << "\t--tole, Tolerance in iterative SVD (default: 1e-5)\n";
	std::cerr << "\t--maf_tole, Tolerance in EM algorithm for allele frequencies (default: 1e-4)\n";
	std::cerr << "\t--hwe_tole, Tolerance in EM algorithm for inbreeding coefficients (default: 1e-4)\n";
}

/* PCAngsd C++ version
Depends on BLAS, LAPACK, OpenMP and Eigen.
*/
int main(int argc, char* argv[]) {
    std::cout << "PCAngsd - Experimental C++ version.\n";
	if (argc == 1) {
		std::cerr << "No argument passed!\n";
		info();
		exit(0);
	}

    // Initialize arguments and default values
    const char* beagle;
    int k = 2;
	int iter = 100;
	int maf_iter = 200;
	int hwe_iter = 200;
	int power = 7;
	int n_threads = 1;
    double maf = 0.05;
	double tole = 1e-5;
	double maf_tole = 1e-4;
	double hwe_tole = 1e-4;
	bool hwe = false;
	bool selection = false;
	bool loadings = false;
	bool maf_save = false;
    std::string out = "pcangsd";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--beagle") == 0) && argv[i+1]) {
            beagle = argv[++i];
        } else if ((strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--n_eig") == 0) && argv[i+1]) {
            k = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) && argv[i+1]) {
            out = argv[++i];
		} else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && argv[i+1])  {
			n_threads = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--hwe") == 0) {
			hwe = true;
		} else if (strcmp(argv[i], "--selection") == 0) {
			selection = true;
		} else if (strcmp(argv[i], "--loadings") == 0) {
			loadings = true;
        } else if (strcmp(argv[i], "--iter") == 0 && argv[i+1]) {
            iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--maf_iter") == 0 && argv[i+1]) {
            maf_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hwe_iter") == 0 && argv[i+1]) {
            hwe_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--power") == 0 && argv[i+1]) {
            power = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--maf") == 0 && argv[i+1]) {
            maf = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tole") == 0 && argv[i+1]) {
            tole = atof(argv[++i]);
        } else if (strcmp(argv[i], "--maf_tole") == 0 && argv[i+1]) {
            maf_tole = atof(argv[++i]);
        } else if (strcmp(argv[i], "--hwe_tole") == 0 && argv[i+1]) {
			hwe_tole = atof(argv[++i]);
		} else if (strcmp(argv[i], "--maf_save") == 0) {
			maf_save = true;
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			info();
			exit(0);
		} else {
            std::cerr << "Unknown argument! " << argv[i] << "\n";
			info();
            exit(-1);
        }
    }

	// Set number of threads globally
	omp_set_num_threads(n_threads);
	std::cout << "Using " << n_threads << " thread(s)\n";

    // Read beagle file to vector
	std::cout << "Parsing Beagle file.\n";
	const char* delims = "\t \n";
	gzFile fp = gzopen(beagle, "r");
	char buf[1000000];
	gzgets(fp, buf, 1000000);
    strtok(buf, delims);
    int c = 1;
    while (strtok(NULL, delims)) {
        c++;
    }
    c -= 3;
	if ((c % 3) != 0) {
		std::cerr << "Incomplete beagle file!\n";
		exit(-1);
	}
	int n = c/3; // Number of individuals
	std::vector<char*> tmp; // Container for all lines
	while (gzgets(fp, buf, 1000000)) {
		tmp.push_back(strdup(buf));
	}
	int m = tmp.size(); // Number of sites
	std::cout << "Parsed " << m << " sites, and " << n << " individuals.\n";

	// Load data into array
	float* L = new float[m*n*3];
	readBeagle(tmp, L, m, n);

    // Estimate allele frequencies
	float* f = new float[m];
    emFrequencies(L, f, m, n, maf_iter, maf_tole);
	if (maf_save) {
		std::ofstream mafvec(out + ".freq");
		if (mafvec.is_open()) {
			for (int j = 0; j < m; j++) {
				mafvec << f[j] << "\n";
			}
		}
	}

    // Filter based on MAF
    if (maf > 0.0) {
        m = filterArrays(L, f, m, n, maf);
        std::cout << m << " sites retained after filtering.\n";
    }

    // Initiate matrices for iterative PCA (SVD)
    MatrixXf P(m, n);
    MatrixXf C(n, n);

    // Estimate individual allele frequencies and GRM
    pcangsdAlgo(L, f, P, C, m, n, k, power, iter, tole);

    // Save GRM
    std::ofstream outmat(out + ".cov");
    if (outmat.is_open()) {
        outmat << C << "\n";
    }

	// Per-site inbreeding estimation
	if (hwe) {
		// Initiate arrays
		float* ind = new float[m];
		float* lrt = new float[m];

		// Run EM algorithm
		pcangsdHWE(L, f, P, ind, lrt, m, n, hwe_iter, hwe_tole);

		// Save outputs
		std::ofstream indvec(out + ".inbreed");
		if (indvec.is_open()) {
			for (int j = 0; j < m; j++) {
				indvec << ind[j] << "\n";
			}
		}
		std::ofstream lrtvec(out + ".lrt");
		if (lrtvec.is_open()) {
			for (int j = 0; j < m; j++) {
				lrtvec << lrt[j] << "\n";
			}
		}
		delete [] ind;
		delete [] lrt;
	}

	// Selection scan (Galinsky)
	if (selection) {
		// Initiate array
		float* d = new float[m*k];

		// Run selection scan
		galinskyScan(L, f, P, d, m, n, k, power);

		// Save output
		std::ofstream selmat(out + ".selection");
		if (selmat.is_open()) {
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < (k-1); i++) {
					selmat << d[j*n + i] << "\t";
				}
				selmat << d[j*n + (k-1)] << "\n";
			}
		}
		delete [] d;
	}

	// SNP loadings
	if (loadings) {
		// Initiate array
		float* w = new float[m*k];

		// Estimate loadings
		snpLoadings(L, f, P, w, m, n, k, power);

		// Save output
		std::ofstream snpmat(out + ".loadings");
		if (snpmat.is_open()) {
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < (k-1); i++) {
					snpmat << w[j*n + i] << "\t";
				}
				snpmat << w[j*n + (k-1)] << "\n";
			}
		}
		delete [] w;
	}
    // Free arrays
    delete [] L;
    delete [] f;
    return 0;
}
