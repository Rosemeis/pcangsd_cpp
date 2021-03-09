void emFrequencies(double* L, double* f, int m, int n, int iter = 200, \
                    double tole = 1e-4);
void initialE(double* L, double* f, Eigen::MatrixXd &E, int m, int n);
void centerE(double* L, double* f, Eigen::MatrixXd &E, \
                Eigen::MatrixXd &P, int m, int n);
void standardE(double* L, double* f, Eigen::MatrixXd &E, Eigen::MatrixXd &P, \
                Eigen::VectorXd &c_diag, int m, int n);
void halkoSVD(Eigen::MatrixXd &E, Eigen::MatrixXd &U, Eigen::VectorXd &s, \
                Eigen::MatrixXd &V, int k, int m, int n);
void pcangsdAlgo(double* L, double* f, Eigen::MatrixXd &E, Eigen::MatrixXd &P, \
                    Eigen::MatrixXd &C, int m, int n, int k, int power = 7, \
                    int iter = 100, double tole = 1e-5);
