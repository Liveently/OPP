#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define EPSILON 0.00001
#define TAU  0.00001

#define N 1024


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
    double y_squared_norm = 0;


    int not_finished = 1;

    while (not_finished) {
        double* u = (double*) calloc(N, sizeof(double));
        double* y = (double*) calloc(N, sizeof(double));
        double* v = (double*) calloc(N, sizeof(double));
        // Ax
#pragma omp parallel for private(j, offset)
        for (int i = 0; i < N; i++) {
            offset = N * i;
            for (j = 0; j < N; j++) {
                u[i] += matrix[offset + j] * result[j];
            }
        }

        // Ax - b
#pragma parallel for
        for (int i = 0; i < N; i++) {
            y[i] = u[i] - b[i];
        }

        y_squared_norm = calc_norm_square(y, N);


        if (y_squared_norm/vector_squared_norm < EPSILON*EPSILON) {
            free(u);
            free(y);
            free(v);
            not_finished = 0;
        }

        if (not_finished) {
            // t(Ax - b)
#pragma parallel for
            for (int i = 0; i < N; i++) {
                v[i] = TAU * y[i];
            }
            // x - t(Ax - b)
#pragma parallel for
            for (int i = 0; i < N; i++) {
                result[i] = result[i] - v[i];
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

	print_vector(result);

	printf("%f\n", time_end - time_start);

	free(matrix);
	free(vector);
	free(result);

	return 0;
}

