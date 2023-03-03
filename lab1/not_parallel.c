#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 5000

double *multiplying_matrixes(double **A, const double *b) {
    double *final = (double *) malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * b[i];
        }
        final[i] = sum;
    }
    return final;
}

double* multuplying_by_scalar(const double* A, double t) {
    double *final = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        final[i] = A[i] * t;
    }
    return final;
}

double* matrixes_subtraction (const double* A, const double* b) {
    double *final = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        final[i] = A[i] - b[i];
    }
    return final;
}

double module(double* a) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

int main() {
    struct timespec start, end;

    double **A = (double **) malloc(N * sizeof(double *));

    for (int i = 0; i < N; i++) {
        A[i] = (double *) malloc(N * sizeof(double));
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j)
                A[i][j] = 1.0;
            else
                A[i][j] = 2.0;
        }
    }

    double *b = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
    }

    double *x = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        x[i] = 0;
    }


    double e = 0.00001;
    double t = 0.001;

    clock_gettime(CLOCK_MONOTONIC, &start);

    while (module(matrixes_subtraction(multiplying_matrixes(A, x), b)) / module(b) > e) {
        x = matrixes_subtraction(x, multuplying_by_scalar(matrixes_subtraction(multiplying_matrixes(A,x), b), t));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));

    for (int i = 0; i < N; i++) {
      //  printf("%f\n", x[i]);
    }
    return 0;
}
