#include <iostream>
#include <vector>
#include <string.h>
#include <zlib.h>

// Read Beagle file into array from vector (inspired by ANGSD)
void readBeagle(std::vector<char*> tmp, float* L, int m, int n) {
	int c = 3*n;
	const char* delims = "\t \n";

    // Read vector file (line by line)
    for (int j = 0; j < m; j++) {
        strtok(tmp[j], delims); // id
        strtok(NULL, delims); // major
        strtok(NULL, delims); // minor
        for (int i = 0; i < c; i++) {
            L[j*c+i] = atof(strtok(NULL, delims));
        }
		free(tmp[j]);
	}
}

// Filter arrays (fake-shrinking)
int filterArrays(float* L, float *f, int m, int n, double tole) {
    std::cout << "Filtering arrays based on MAF threshold: " << tole << ".\n";
    int c = 0;
    for (int j = 0; j < m; j++) {
        if (f[j] > tole && f[j] < (1.0 - tole)) {
            if (c != j) {
                f[c] = f[j];
                for (int i = 0; i < n*3; i++) {
                    L[c*3*n+i] = L[j*3*n+i];
                }
            }
            c++;
        }
    }
    return c;
}
