#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N1 1000
#define N2 1000
#define N3 1000

#define P1 2
#define P2 2

double **multiplying_matrixes(double **A, double **B, int overall_frame_y, int overall_frame_x) {
    double **C_sub = (double **) malloc(overall_frame_y * sizeof(double));

    for (int i = 0; i < overall_frame_y; i++) {
        C_sub[i] = (double *) malloc(overall_frame_x * sizeof(double));
    }

    for (int i = 0; i < overall_frame_y; i++) {
        for (int k = 0; k < overall_frame_x; k++) {
            double sum = 0;
            for (int j = 0; j < N2; j++) {
                sum += A[i][j] * B[j][k];
            }
            C_sub[i][k] = sum;
        }
    }
    return C_sub;
}

int main(int argc, char *argv[]) {

    double start_time;
    double finish_time;

    MPI_Init(&argc, &argv);

    int dims[2] = {P1, P2}, periods[2] = {0, 0}, coords[2], reorder = 1;
    int size, rank;

    MPI_Comm comm2d;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(size, 2, dims); //Создает разделение процессоров в декартовой сетке

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d); //создание доп комутатора, с подключением к декартовой сетки
    MPI_Comm_rank(comm2d, &rank);

    MPI_Cart_get(comm2d, 2, dims, periods, coords); //получаем всю информацию о топологии

    double **A = (double **) malloc(N1 * sizeof(double *));
    for (int i = 0; i < N1; i++) {
        A[i] = (double *) malloc(N2 * sizeof(double));
    }

    double **B = (double **) malloc(N2 * sizeof(double *));
    for (int i = 0; i < N2; i++) {
        B[i] = (double *) malloc(N3 * sizeof(double));
    }

    if (rank == 0) {
        for (int i = 0; i < N1; i++)
            for (int j = 0; j < N2; j++)
                    A[i][j] = rand() % 10;


        for (int i = 0; i < N2; i++)
            for (int j = 0; j < N3; j++)
                    B[i][j] = rand() % 10;


    }

    start_time = MPI_Wtime();

    //Во всех ветвях задаем подматрицы
    //Предполагается что деление без остатка

    int overall_frame_y = N1 / P1; //кол-во строк
    int overall_frame_x = N3 / P2; //кол-во столбцов

    double **A_overall = (double **) malloc(overall_frame_y * sizeof(double *));
    for (int i = 0; i < overall_frame_y; i++) {
        A_overall[i] = (double *) malloc(N2 * sizeof(double));
    }

    double **B_overall = (double **) malloc(N2 * sizeof(double *));
    for (int i = 0; i < N2; i++) {
        B_overall[i] = (double *) malloc(overall_frame_x * sizeof(double));
    }

    if (rank == 0) { //здесь получаем кусочку матрицы для всех процессов и рассылаем

        for (int y = P1 * P2-1; y >=0 ; y--) {
            int divided_i = y / P2;
            int divided_j = y % P2;

            for (int i = 0; i < overall_frame_y; i++) {
                for (int j = 0; j < N2; j++) {
                    A_overall[i][j] = A[divided_i * overall_frame_y + i][j];
                }
            }

            for (int i = 0; i < N2; i++) {
                for (int j = 0; j < overall_frame_x; j++) {
                    B_overall[i][j] = B[i][overall_frame_x * divided_j + j];
                }
            }

            for (int i = 0; i < overall_frame_y; i++) {
                MPI_Send(A_overall[i], N2, MPI_DOUBLE, y, i, comm2d);
            }

            for (int i = 0; i < N2; i++) {
                MPI_Send(B_overall[i], overall_frame_x, MPI_DOUBLE, y, i, comm2d);
            }
        }
        
    }


    for (int y = 1; y < P1 * P2; y++) {  //получаем кусочек матриц
        if (rank == y) {
            for (int i = 0; i < overall_frame_y; i++) {
                MPI_Recv(A_overall[i], N2, MPI_DOUBLE, 0, i, comm2d, MPI_STATUS_IGNORE); //Выполняет операцию получения
            }
            for (int i = 0; i < N2; i++) {
                MPI_Recv(B_overall[i], overall_frame_x, MPI_DOUBLE, 0, i, comm2d, MPI_STATUS_IGNORE); //Выполняет операцию получения
            }
        }
    }


    int amount = overall_frame_y * overall_frame_x;
    double *C_all = (double *) malloc(amount * P1 * P2 * sizeof(double)); // полная матрица

    double **C_overall = (double **) malloc(N1 * sizeof(double *)); //тоже полная матрица
    for (int i = 0; i < N1; i++) {
        C_overall[i] = (double *) malloc(N3 * sizeof(double));
    }

    double **C_sub = multiplying_matrixes(A_overall, B_overall, overall_frame_y, overall_frame_x);  //промежуточный результат умножения кусочков

    int k = 0;
    double *C = (double *) malloc(amount * sizeof(double)); //тоже самое что C_sub
    for (int i = 0; i < overall_frame_y; i++) {
        for (int j = 0; j < overall_frame_x; j++) {
            C[k] = C_sub[i][j];
            k++;
        }
    }

    MPI_Gather(C, amount, MPI_DOUBLE, C_all, amount, MPI_DOUBLE, 0, comm2d); //Собираем данные от всех участников группы к одному участнику собираем C в С_all
    int p = 0;
    int r = 0;
    if (rank == 0) {
        while (r != P1 * P2) {
            double *part = (double *) malloc(amount * sizeof(double));
            for (int i = 0; i < amount; i++) {
                part[i] = C_all[p];
                p++;
            }
            int divided_i = r / P2;
            int divided_j = r % P2;

            int u = 0;
            for (int i = 0; i < overall_frame_y; i++) {
                for (int j = 0; j < overall_frame_x; j++) {
                    C_overall[i + overall_frame_y * divided_i][j + overall_frame_x * divided_j] = part[u];
                    u++;
                }
            }
            r++;
        }
    }

    finish_time = MPI_Wtime();
    

    if (rank == 0) {

        int er=0;

        double **m_ = multiplying_matrixes(A, B, N1, N3);

        for (int i = 0; i < N1; i++)
            for (int j = 0; j < N3; j++)
                if (C_overall[i][j] != m_[i][j]) er=1;


        (er==0)?  printf("the calculation is correct \n") :  printf("the calculation is incorrect \n");
        printf("Time: %lf sec\n", finish_time - start_time);

    }


    MPI_Finalize();
    return 0;
}
