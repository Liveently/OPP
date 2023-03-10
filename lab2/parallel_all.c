#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define EPSILON 0.00001
#define TAU  0.00001

#define N 1024

int THREADS;

double* create_matrix() {
    double* chunk = (double*) calloc(N*N, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            chunk[i * N + j] = (i == j) ? 2 : 1;
        }
    }
    return chunk;
}

double* create_b() {
    double* vector = (double*) calloc(N, sizeof(double));
    for (int i = 0; i < N; i++) {
        vector[i] = N + 1;
    }
    return vector;
}


double* calculate(const double* matrix, const double* vector) {
    int j;
    double* result = (double*) calloc(N, sizeof(double));
    double vector_squared_norm = 0;
    double Axb_squared_norm = 0;
    double* Ax;
    double* Axb;
    double* v;
    for (int i = 0; i < N; i++) {
        vector_squared_norm += vector[i] * vector[i];
    }
    int not_finished = 1;
#pragma omp parallel private(j) shared(not_finished, y_squared_norm)
    {
        while (not_finished) {
#pragma omp single
            {
                Ax = (double*) calloc(N, sizeof(double));
                Axb = (double*) calloc(N, sizeof(double));
                v = (double*) calloc(N, sizeof(double));
                Axb_squared_norm = 0;
            }
            // Ax
#pragma omp for schedule(static, N/THREADS)
            for (int i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    Ax[i] += matrix[N * i + j] * result[j];
                }
            }

            // Ax - b
#pragma omp for schedule(static, N/THREADS)
            for (int i = 0; i < N; i++) {
                Axb[i] = Ax[i] - vector[i];
            }

#pragma parallel for reduction(+:y_squared_norm)
            for (int i = 0; i < N; i++) {
                Axb_squared_norm += Axb[i] * Axb[i];
            }
#pragma omp single
            if (Axb_squared_norm / vector_squared_norm < EPSILON * EPSILON) {
                free(Ax);
                free(Axb);
                free(v);
                not_finished = 0;
            }

            if (not_finished) {
                // t(Ax - b)
#pragma omp for schedule(static, N/THREADS)
                for (int i = 0; i < N; i++) {
                   v [i] = TAU * Axb[i];
                }
                // x - t(Ax - b)
#pragma omp for schedule(static, N/THREADS)
                for (int i = 0; i < N; i++) {
                    result[i] = result[i] - v[i];
                }
            }
        }
    }
    return result;
}

int main(int argc, char* argv[]) {

    double* matrix = create_matrix();
    double* vector = create_b();
    double* result;

    double time_start, time_end;
    THREADS = omp_get_num_threads();
    time_start = omp_get_wtime();
    result = calculate(matrix, vector);
    time_end = omp_get_wtime();

    double norm_square = 0.0;
    for (int i = 0; i < N; ++i)
        norm_square += result[i] * result[i];

    printf("Norm sqrt: %lf\n", norm_square);
    printf("Time: %lf sec\n", time_end - time_start);


    free(matrix);
    free(vector);
    free(result);

    return 0;
}
