#include <iostream>
#include <cmath>
#include <armadillo>
#include "omp.h"

// Estimate allele frequencies
void emFrequencies(double* l, double* f, int m, int n, int iter, double tole) {
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
                p0 = l[j*3*n+3*i+0]*(1.0 - f[j])*(1.0 - f[j]);
                p1 = l[j*3*n+3*i+1]*2*f[j]*(1.0 - f[j]);
                p2 = l[j*3*n+3*i+2]*f[j]*f[j];
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
void initialE(double* l, double* f, arma::mat &e, int m, int n) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double p0, p1, p2;
        for (int i = 0; i < n; i++) {
            p0 = l[j*3*n+3*i+0]*(1.0 - f[j])*(1.0 - f[j]);
            p1 = l[j*3*n+3*i+1]*2*f[j]*(1.0 - f[j]);
            p2 = l[j*3*n+3*i+2]*f[j]*f[j];
            e(j,i) = (p1 + 2*p2)/(p0 + p1 + p2) - 2.0*f[j];
        }
    }
}

// Center E with individual allele frequencies
void centerE(double* l, double* f, arma::mat &e, \
            arma::mat &p, int m, int n) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double p0, p1, p2;
        for (int i = 0; i < n; i++) {
            // Rescale individual allele frequencies
            p(j,i) = (p(j,i) + 2.0*f[j])/2.0;
            p(j,i) = std::fmin(std::fmax(p(j,i), 1e-4), 1.0 - (1e-4));

            // Update e
            p0 = l[j*3*n+3*i+0]*(1.0 - p(j,i))*(1.0 - p(j,i));
            p1 = l[j*3*n+3*i+1]*2*p(j,i)*(1.0 - p(j,i));
            p2 = l[j*3*n+3*i+2]*p(j,i)*p(j,i);
            e(j,i) = (p1 + 2*p2)/(p0 + p1 + p2) - 2.0*f[j];
        }
    }
}

// Standardize E with individual allele frequencies
void standardE(double* l, double* f, arma::mat &e, arma::mat &p, \
                arma::vec &diag_c, int m, int n) {
    #pragma omp parallel
    {
        double diag_private[n] = {0}; // Thread private array
        #pragma omp for
        for (int j = 0; j < m; j++) {
            double p0, p1, p2, pSum, tmp;
            double norm = sqrt(2.0*f[j]*(1.0 - f[j]));
            for (int i = 0; i < n; i++) {
                // Rescale individual allele frequencies
                p(j,i) = (p(j,i) + 2.0*f[j])/2.0;
                p(j,i) = std::fmin(std::fmax(p(j,i), 1e-4), 1.0 - (1e-4));

                // Update e
                p0 = l[j*3*n+3*i+0]*(1.0 - p(j,i))*(1.0 - p(j,i));
                p1 = l[j*3*n+3*i+1]*2*p(j,i)*(1.0 - p(j,i));
                p2 = l[j*3*n+3*i+2]*p(j,i)*p(j,i);
                pSum = p0 + p1 + p2;
                e(j,i) = (p1 + 2*p2)/pSum - 2.0*f[j];
                e(j,i) = e(j,i)/norm;

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

// Halko SVD
void halkoSVD(arma::mat &e, arma::mat &u, arma::vec &s, arma::mat &v, int k, \
                int power, int m, int n) {
    int t = k + 10;
    arma::mat M1 = arma::randn<arma::mat>(n, t);
    arma::mat M2(m, t), M3(m, t), MT(t, t), B(t, n), Uhat(m, t);
    arma::vec S(t);
    M2 = e * M1;
    for (int it = 0; it < power; it++) {
        arma::lu(M3, MT, M2);
        M1 = e.t() * M3;
        M2 = e * M1;
    }
    arma::qr_econ(M3, MT, M2);
    B = M3.t() * e;
    arma::svd_econ(MT, S, M1, B);
    Uhat = M3 * MT;

    // Update old arrays
    u = Uhat.cols(0,k);
    s = S.rows(0,k);
    v = M1.cols(0,k);
}

// Iterative PCA
void pcangsdAlgo(double* l, double* f, arma::mat &e, arma::mat &p, \
                    arma::mat &u, arma::vec &s, arma::mat &v, arma::mat &c, \
                    int m, int n, int k, int power, int iter, double tole) {
    std::cout << "\nEstimating individual allele frequencies.\n";
    double diff;
    double flip;
    arma::mat v_prev(k, n);
    arma::vec c_diag(n);

    // Initialize e and estimate SVD
    initialE(l, f, e, m, n);
    halkoSVD(e, u, s, v, k, power, m, n);
    p = u * (diagmat(s) * v.t());
    std::cout << "Individual allele frequencies estimated (1).\n";

    // Run iterative updates
    for (int it = 0; it < iter; it++) {
        v_prev = v; // Previous right singular vectors
        centerE(l, f, e, p, m, n);
        halkoSVD(e, u, s, v, k, power, m, n);
        p = u * (diagmat(s) * v.t());

        // Calculate differences between iterations
        diff = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                if (arma::sign(v(i,j)) == arma::sign(v_prev(i,j))) {
                    diff += (v(i,j) - v_prev(i,j))*(v(i,j) - v_prev(i,j));
                } else {
                    diff += (v(i,j) + v_prev(i,j))*(v(i,j) + v_prev(i,j));
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
    standardE(l, f, e, p, c_diag, m, n);
    c = e.t() * e;
    c /= (double)m;
    c.diag() = c_diag/(double)m;
}
