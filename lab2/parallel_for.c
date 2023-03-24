#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define EPSILON 0.00001
#define TAU  0.00001

#define N 7000


double* create_matrix() {
    double* chunk = (double*) calloc(N*N, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            chunk[i * N + j] = (i == j) ? 2 : 1;
        }
    }
    return chunk;
}

double* create_vector() {
    double* vector = (double*) calloc(N, sizeof(double));
    for (int i = 0; i < N; i++) {
        vector[i] = N + 1;
    }
    return vector;
}

void print_vector(double* vector) {
    for (int i = 0; i < N; i++) {
        printf("%0.4f\n", vector[i]);
    }
}

double calc_norm_square(const double* vector, int size)
{
    double norm_square = 0.0;

#pragma omp parallel for \
                     reduction(+ : norm_square)
    for (int i = 0; i < size; ++i)
        norm_square += vector[i] * vector[i];

    return norm_square;
}

double* calculate(const double* matrix, const double* b) {

    int j, offset;
    double* result = (double*) calloc(N, sizeof(double));
    double vector_squared_norm = calc_norm_square(b, N);
    double Axb_squared_norm = 0;


    int not_finished = 1;


    double* Ax = (double*) calloc(N, sizeof(double));
    double* Axb = (double*) calloc(N, sizeof(double));
    double* v = (double*) calloc(N, sizeof(double));


    while (not_finished) {

        // Ax
#pragma omp parallel for private(j, offset)
        for (int i = 0; i < N; i++) {
            offset = N * i;
            for (j = 0; j < N; j++) {
                Ax[i] += matrix[offset + j] * result[j];
            }
        }

        // Ax - b
#pragma parallel for
        for (int i = 0; i < N; i++) {
            Axb[i] = Ax[i] - b[i];
        }

        Axb_squared_norm = calc_norm_square(Axb, N);


        if (Axb_squared_norm / vector_squared_norm < EPSILON * EPSILON) {
            free(Ax);
            free(Axb);
            free(v);
            not_finished = 0;
        }

        if (not_finished) {
            // t(Ax - b)
#pragma parallel for
            for (int i = 0; i < N; i++) {
                v[i] = TAU * Axb[i];
            }
            // x - t(Ax - b)
#pragma parallel for
            for (int i = 0; i < N; i++) {
                result[i] = result[i] - v[i];
            }

            for (int i = 0; i < N; i++) {
                Ax[i] = 0;
                Axb[i] = 0;
                v[i] = 0;
            }

        }
    }
    return result;
}

int main(int argc, char* argv[]) {

    double* matrix = create_matrix();
    double* vector = create_vector();
    double* result;

    double time_start, time_end;
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
