#ifndef SHARED_H
#define SHARED_H
#include <Eigen/Dense>

void pcangsdHWE(float* L, float* f, Eigen::MatrixXf &P, float* ind, \
				float* lrt, int m, int n, int iter, double tole);
void standardSelection(float* L, float* f, Eigen::MatrixXf &E, \
				Eigen::MatrixXf &P, int m, int n);
void galinskyScan(float* L, float* f, Eigen::MatrixXf &P, float* d, int m, \
				int n, int k, int power);
void snpLoadings(float* L, float*f, Eigen::MatrixXf &P, float* w, int m, \
				int n, int k, int power);

#endif
