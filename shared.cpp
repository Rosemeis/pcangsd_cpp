#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "omp.h"
#include "pca.hpp"
using namespace Eigen;

// Estimate per-site inbreeding coefficients
void pcangsdHWE(float* L, float* f, MatrixXf &P, float* ind, float* lrt, \
				int m, int n, int iter, double tole) {
	std::cout << "\nEstimating per-site inbreeding coefficients\n";
	double diff;

    // Initialize temp array as well as arrays
    float* ind_prev = new float[m];
    for (int j = 0; j < m; j++) {
        ind[j] = 0.0;
    }

	// Run EM algorithm
	for (int it = 0; it < iter; it++) {
		// Define multithreaded loop
		#pragma omp parallel for
		for (int j = 0; j < m; j++) {
			ind_prev[j] = ind[j];
			double fAdj, tmp0, tmp1, tmp2, tmpSum;
			double expH = 0.0, p0 = 0.0, p1 = 0.0, p2 = 0.0;
			for (int i = 0; i < n; i++) {
				fAdj = (1.0 - P(j,i))*P(j,i)*ind[j];
				tmp0 = std::fmax((1.0 - P(j,i))*(1.0 - P(j,i)) + fAdj, 1e-4);
				tmp1 = std::fmax((1.0 - P(j,i))*P(j,i) - 2*fAdj, 1e-4);
				tmp2 = std::fmax(P(j,i)*P(j,i) + fAdj, 1e-4);
				tmpSum = tmp0 + tmp1 + tmp2;

				// Readjust distribution
				tmp0 = tmp0/tmpSum;
				tmp1 = tmp1/tmpSum;
				tmp2 = tmp2/tmpSum;

				// Posterior
				tmp0 = L[j*3*n+3*i+0]*tmp0;
				tmp1 = L[j*3*n+3*i+1]*tmp1;
				tmp2 = L[j*3*n+3*i+2]*tmp2;
				tmpSum = tmp0 + tmp1 + tmp2;

				// Sum over individuals
				p0 = p0 + tmp0/tmpSum;
				p1 = p1 + tmp1/tmpSum;
				p2 = p2 + tmp2/tmpSum;

				// Count heterozygotes
				expH = expH + 2*(1.0 - P(j,i))*P(j,i);
			}
			// ANGSD procedure
			p0 = std::fmax(p0/(double)n, 1e-4);
			p1 = std::fmax(p1/(double)n, 1e-4);
			p2 = std::fmax(p2/(double)n, 1e-4);

			// Update the inbreeding coefficient
			ind[j] = 1.0 - ((double)n * p1/expH);
			ind[j] = std::fmin(std::fmax(-1.0, ind[j]), 1.0);
		}
		// Calculate differences between iterations
		diff = 0.0;
		for (int j = 0; j < m; j++) {
			diff += (ind[j] - ind_prev[j])*(ind[j] - ind_prev[j]);
		}
		diff = sqrt(diff/(double)m);

		// Check for convergence
        if (diff < tole) {
            std::cout << "EM (HWE) converged at iteration: " << it+1 << "\n";
            break;
        } else if (it == (iter-1)) {
            std::cout << "EM (HWE) did not converge.\n";
        }
	}
	// Estimate log-likelihoods
	#pragma omp parallel for
	for (int j = 0; j < m; j++) {
		double fAdj, tmp0, tmp1, tmp2, tmpSum;
		double logA = 0.0, logN = 0.0;
		for (int i = 0; i < n; i++) {
			// Alternative
			fAdj = (1.0 - P(j,i))*P(j,i)*ind[j];
			tmp0 = std::fmax((1.0 - P(j,i))*(1.0 - P(j,i)) + fAdj, 1e-4);
			tmp1 = std::fmax((1.0 - P(j,i))*P(j,i) - 2*fAdj, 1e-4);
			tmp2 = std::fmax(P(j,i)*P(j,i) + fAdj, 1e-4);
			tmpSum = tmp0 + tmp1 + tmp2;

			// Readjust distribution
			tmp0 = tmp0/tmpSum;
			tmp1 = tmp1/tmpSum;
			tmp2 = tmp2/tmpSum;

			// Likelihood
			tmp0 = L[j*3*n+3*i+0]*tmp0;
			tmp1 = L[j*3*n+3*i+1]*tmp1;
			tmp2 = L[j*3*n+3*i+2]*tmp2;
			logA = logA + log(tmp0 + tmp1 + tmp2);

			// Null model
			tmp0 = L[j*3*n+3*i+0]*(1.0 - P(j,i))*(1.0 - P(j,i));
			tmp1 = L[j*3*n+3*i+1]*2*(1.0 - P(j,i))*P(j,i);
			tmp2 = L[j*3*n+3*i+2]*P(j,i)*P(j,i);
			logN = logN + log(tmp0 + tmp1 + tmp2);
		}
		lrt[j] = 2*(logA - logN);
	}
	// Release memory
	delete [] ind_prev;
}

// Standardize E for selection or loadings
void standardSelection(float* L, float* f, MatrixXf &E, MatrixXf &P, \
						int m, int n) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double p0, p1, p2;
		double norm = sqrt(2.0*f[j]*(1.0 - f[j]));
        for (int i = 0; i < n; i++) {
            // Update e
            p0 = L[j*3*n+3*i+0]*(1.0 - P(j,i))*(1.0 - P(j,i));
            p1 = L[j*3*n+3*i+1]*2*P(j,i)*(1.0 - P(j,i));
            p2 = L[j*3*n+3*i+2]*P(j,i)*P(j,i);
            E(j,i) = (p1 + 2*p2)/(p0 + p1 + p2) - 2.0*f[j];
			E(j,i) = E(j,i)/norm;
        }
    }
}

// Galinsky selection scan
void galinskyScan(float* L, float* f, MatrixXf &P, float* d, int m, int n, \
					int k, int power) {
	// Initialize matrices and vectors for SVD
	MatrixXf E(m, n), U(m, k), V(n, k);
	VectorXf s(k);
	standardSelection(L, f, E, P, m, n);

	// Perform SVD
	halkoSVD(E, U, s, V, k, power, m, n);

	// Estimate test statistics
	#pragma omp parallel for
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < k; i++) {
			d[j*n + i] = U(j,i)*U(j,i)*(double)m;
		}
	}
}

// Estimate SNP loadings
void snpLoadings(float* L, float*f, MatrixXf &P, float* w, int m, int n, \
					int k, int power) {
	// Initialize matrices and vectors for SVD
	MatrixXf E(m, n), U(m, k), V(n, k);
	VectorXf s(k);
	standardSelection(L, f, E, P, m, n);

	// Perform SVD
	halkoSVD(E, U, s, V, k, power, m, n);

	// Estimate test statistics
	#pragma omp parallel for
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < k; i++) {
			w[j*n + i] = U(j,i)*s(i)*s(i)/(double)m;
		}
	}
}
