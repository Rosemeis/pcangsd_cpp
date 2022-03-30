#ifndef PCA_H
#define PCA_H
#include <Eigen/Dense>

void emFrequencies(float* L, float* f, int m, int n, int iter, double tole);
void initialE(float* L, float* f, Eigen::MatrixXf &E, int m, int n);
void centerE(float* L, float* f, Eigen::MatrixXf &E, Eigen::MatrixXf &P, \
				int m, int n);
void standardE(float* L, float* f, Eigen::MatrixXf &E, Eigen::MatrixXf &P, \
				Eigen::VectorXf &c_diag, int m, int n);
void halkoSVD(Eigen::MatrixXf &E, Eigen::MatrixXf &U, Eigen::VectorXf &s, \
				Eigen::MatrixXf &V, int k, int power, int m, int n);
void pcangsdAlgo(float* L, float* f, Eigen::MatrixXf &P, Eigen::MatrixXf &C, \
				int m, int n, int k, int power, int iter, double tole);

#endif
