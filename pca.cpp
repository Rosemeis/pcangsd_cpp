#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include <iostream>
#include <cmath>
#include <random>
#include "omp.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
using namespace Eigen;

// Estimate allele frequencies
void emFrequencies(double* L, double* f, int m, int n, int iter, double tole) {
    std::cout << "\nEstimating allele frequencies.\n";
    double diff;

    // Initialize temp array as well as freq array
    double* f_prev = new double[m];
    for (int j = 0; j < m; j++) {
        f[j] = 0.25;
    }

    // Run EM algorithm
    for (int it = 0; it < iter; it++) {
        // Define multithreaded loop
        #pragma omp parallel for
        for (int j = 0; j < m; j++) {
            f_prev[j] = f[j];
            double p0, p1, p2, tmp = 0.0;
            for (int i = 0; i < n; i++) {
                p0 = L[j*3*n+3*i+0]*(1.0 - f[j])*(1.0 - f[j]);
                p1 = L[j*3*n+3*i+1]*2*f[j]*(1.0 - f[j]);
                p2 = L[j*3*n+3*i+2]*f[j]*f[j];
                tmp = tmp + (p1 + 2.0*p2)/(2*(p0 + p1 + p2));
            }
            f[j] = tmp/(double)n;
        }
        // Calculate differences between iterations
        diff = 0.0;
        for (int j = 0; j < m; j++) {
            diff += (f[j] - f_prev[j])*(f[j] - f_prev[j]);
        }
        diff = sqrt(diff/(double)m);

        // Check for convergence
        if (diff < tole) {
            std::cout << "EM (MAF) converged at iteration: " << it+1 << "\n";
            break;
        } else if (it == (iter-1)) {
            std::cout << "EM (MAF) did not converge.\n";
        }
    }
    delete [] f_prev;
}

// Initiate and center E with population allele frequencies
void initialE(double* l, double* f, MatrixXd &E, int m, int n) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double p0, p1, p2;
        for (int i = 0; i < n; i++) {
            p0 = l[j*3*n+3*i+0]*(1.0 - f[j])*(1.0 - f[j]);
            p1 = l[j*3*n+3*i+1]*2*f[j]*(1.0 - f[j]);
            p2 = l[j*3*n+3*i+2]*f[j]*f[j];
            E(j,i) = (p1 + 2*p2)/(p0 + p1 + p2) - 2.0*f[j];
        }
    }
}

// Center E with individual allele frequencies
void centerE(double* L, double* f, MatrixXd &E, \
            MatrixXd &P, int m, int n) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double p0, p1, p2;
        for (int i = 0; i < n; i++) {
            // Rescale individual allele frequencies
            P(j,i) = (P(j,i) + 2.0*f[j])/2.0;
            P(j,i) = std::fmin(std::fmax(P(j,i), 1e-4), 1.0 - (1e-4));

            // Update e
            p0 = L[j*3*n+3*i+0]*(1.0 - P(j,i))*(1.0 - P(j,i));
            p1 = L[j*3*n+3*i+1]*2*P(j,i)*(1.0 - P(j,i));
            p2 = L[j*3*n+3*i+2]*P(j,i)*P(j,i);
            E(j,i) = (p1 + 2*p2)/(p0 + p1 + p2) - 2.0*f[j];
        }
    }
}

// Standardize E with individual allele frequencies
void standardE(double* L, double* f, MatrixXd &E, MatrixXd &P, \
                VectorXd &diag_c, int m, int n) {
    #pragma omp parallel
    {
        double diag_private[n] = {0}; // Thread private array
        #pragma omp for
        for (int j = 0; j < m; j++) {
            double p0, p1, p2, pSum, tmp;
            double norm = sqrt(2.0*f[j]*(1.0 - f[j]));
            for (int i = 0; i < n; i++) {
                // Rescale individual allele frequencies
                P(j,i) = (P(j,i) + 2.0*f[j])/2.0;
                P(j,i) = std::fmin(std::fmax(P(j,i), 1e-4), 1.0 - (1e-4));

                // Update e
                p0 = L[j*3*n+3*i+0]*(1.0 - P(j,i))*(1.0 - P(j,i));
                p1 = L[j*3*n+3*i+1]*2*P(j,i)*(1.0 - P(j,i));
                p2 = L[j*3*n+3*i+2]*P(j,i)*P(j,i);
                pSum = p0 + p1 + p2;
                E(j,i) = (p1 + 2*p2)/pSum - 2.0*f[j];
                E(j,i) = E(j,i)/norm;

                // Update diag
                tmp = (0.0 - 2.0*f[j])*(0.0 - 2.0*f[j])*(p0/pSum);
                tmp = tmp + (1.0 - 2.0*f[j])*(1.0 - 2.0*f[j])*(p1/pSum);
                tmp = tmp + (2.0 - 2.0*f[j])*(2.0 - 2.0*f[j])*(p2/pSum);
                diag_private[i] += tmp/(2.0*f[j]*(1.0 - f[j]));
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                diag_c[i] += diag_private[i]; // Sum arrays for threads
            }
        }
    }
}

// Generate random matrix (standard normal)
void generateRand(MatrixXd &Omg, std::normal_distribution<double> dist, \
                    int n, int t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < t; j++) {
            Omg(i,j) = dist(gen);
        }
    }
}

// Halko SVD
void halkoSVD(MatrixXd &E, MatrixXd &U, VectorXd &s, MatrixXd &V, int k, \
                int power, int m, int n) {
    int t = k + 10;
    MatrixXd Q1(n, t), Q2(m, t), R(t, t), B(t, n);
    std::normal_distribution<double> dist(0, 1);
    generateRand(Q1, dist, n, t);
    Q2 = E * Q1;
    Q1 = E.transpose() * Q2;
    for (int it = 0; it < power; it++) {
        HouseholderQR<MatrixXd> qr(Q1);
        Q1 = qr.householderQ() * MatrixXd::Identity(n, t);
        Q2 = E * Q1;
        Q1 = E.transpose() * Q2;
    }
    HouseholderQR<MatrixXd> qr(Q2);
    R = MatrixXd::Identity(t, m) * qr.matrixQR().triangularView<Upper>();
    B = R.inverse().transpose() * Q1.transpose();
    BDCSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);

    // Update old arrays
    Q2 = qr.householderQ() * MatrixXd::Identity(m, t);
    U = Q2 * svd.matrixU().leftCols(k);
    s = svd.singularValues().head(k);
    V = svd.matrixV().leftCols(k);
}

// Iterative PCA
void pcangsdAlgo(double* L, double* f, MatrixXd &E, MatrixXd &P, \
                    MatrixXd &C, int m, int n, int k, int power, int iter, \
                    double tole) {
    std::cout << "\nEstimating individual allele frequencies.\n";
    double diff;
    double flip;
    MatrixXd U(m, k), V(n, k), V_prev(n, k);
    VectorXd s(k), c_diag(n);

    // Initialize e and estimate SVD
    initialE(L, f, E, m, n);
    halkoSVD(E, U, s, V, k, power, m, n);
    P = U * (s.asDiagonal() * V.transpose());
    std::cout << "Individual allele frequencies estimated (1).\n";

    // Run iterative updates
    for (int it = 0; it < iter; it++) {
        V_prev = V; // Previous right singular vectors
        centerE(L, f, E, P, m, n);
        halkoSVD(E, U, s, V, k, power, m, n);
        P = U * (s.asDiagonal() * V.transpose());

        // Calculate differences between iterations
        diff = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                if (((V(i,j) > 0) - (V(i,j) < 0)) == ((V_prev(i,j) > 0) - (V_prev(i,j) < 0))) {
                    diff += (V(i,j) - V_prev(i,j))*(V(i,j) - V_prev(i,j));
                } else {
                    diff += (V(i,j) + V_prev(i,j))*(V(i,j) + V_prev(i,j));
                }
            }
        }
        diff = sqrt(diff/((double)(k*n)));

        // Check for convergence
        std::cout << "Individual allele frequencies estimated (" << it+2 \
                    << "). RMSE=" << diff << "\n";
        if (diff < tole) {
            std::cout << "PCAngsd converged.\n";
            break;
        } else if (it == (iter-1)) {
            std::cout << "PCAngsd did not converge.\n";
        }
    }
    // Estimate GRM
    std::cout << "Computing GRM.\n";
    standardE(L, f, E, P, c_diag, m, n);
    C = E.transpose() * E;
    C.array() /= (double)m;
    C.diagonal() = c_diag.array()/(double)m;
}
