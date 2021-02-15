#include <iostream>
#include <string.h>
#include <zlib.h>

// Read Beagle file (inspired by ANGSD)
void readBeagle(const char* beagle, double* l, int m, int n) {
    std::cout << "Parsing Beagle file.\n";
    const char* delims = "\t \n";

    // Initiate file pointer
    gzFile fp = gzopen(beagle, "r");
    char buf[1000000];

    // Verify number of individuals specified - first line
    gzgets(fp, buf, 1000000);
    strtok(buf, delims);
    int c = 1;
    while (strtok(NULL, delims))  {
        c++;
    }
    c -= 3;
    if (n != c/3) {
        std::cerr << "\nNumber of specified individuals does not match!\n";
        exit(-1);
    }
    // Read gzipped file, line by line
    int j = 0;
    while (gzgets(fp, buf, 1000000)) {
        strtok(buf, delims); // id
        strtok(NULL, delims); // major
        strtok(NULL, delims); // minor
        for (int i = 0; i < c; i++) {
            l[j*c+i] = atof(strtok(NULL, delims));
        }
        j++;
    }
    // Verify number of sites specified
    if (m != j) {
        std::cerr << "\nNumber of specified sites does not match!\n";
        exit(-1);
    }
    gzclose(fp);
}

// Filter arrays (fake-shrinking)
int filterArrays(double* l, double *f, int m, int n, double tole) {
    std::cout << "Filtering arrays based on MAF threshold: " << tole << ".\n";
    int c = 0;
    for (int j = 0; j < m; j++) {
        if (f[j] > tole && f[j] < (1.0 - tole)) {
            if (c != j) {
                f[c] = f[j];
                for (int i = 0; i < n*3; i++) {
                    l[c*3*n+i] = l[j*3*n+i];
                }
            }
            c++;
        }
    }
    return c;
}
