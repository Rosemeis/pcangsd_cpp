void emFrequencies(double* l, double* f, int m, int n, int iter = 200, \
                    double tole = 1e-4);
void initialE(double* l, double* f, arma::mat &e, int m, int n);
void centerE(double* l, double* f, arma::mat &e, \
                arma::mat &p, int m, int n);
void standardE(double* l, double* f, arma::mat &e, arma::mat &p, \
                arma::vec &c_diag, int m, int n);
void halkoSVD(arma::mat &e, arma::mat &u, arma::vec &s, arma::mat &v, int k, \
                int m, int n);
void pcangsdAlgo(double* l, double* f, arma::mat &e, arma::mat &p, \
                    arma::mat &u, arma::vec &s, arma::mat &v, arma::mat &c, \
                    int m, int n, int k, int power = 7, int iter = 100, \
                    double tole = 1e-5);
