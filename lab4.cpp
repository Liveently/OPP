#include <cstring>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <mpi.h>

using namespace std;

const double eps = 1e-8;
const double a = 1e5;

const int Nx = 256;
const int Ny = 256;
const int Nz = 256;

const double min_x = -1, min_y = -1, min_z = -1;
const double max_x = 1,  max_y = 1,  max_z = 1;


double phi_finction(double x, double y, double z) {
    return x*x+y*y+z*z;
}

double ro(double x, double y, double z) {
    return 6-a*phi_finction(x,y,z);
}

double update_layer(int base_z, int z, double* values, double* tmp_values, double hx, double hy, double hz) {
    int abs_z = base_z+z;

    if (abs_z==0 || abs_z==Nz-1) {  //если верх или низ
        //Копируем этот слой в новый массив на старое место, не пересчитывая
        memcpy(tmp_values + z*Nx*Ny, values + z*Nx*Ny, Nx * Ny * sizeof(double));
        return 0;
    }
    //Иначе пересчитываем каждый элемент слоя с помощью итерационной формулы
    double max_delta = 0;
    double cur_z = min_z+abs_z*hz;
    for (int i=0;i<Nx;i++) {
        double cur_x = min_x+i*hx;
        for (int j=0;j<Ny;j++) {
            double cur_y = min_y+j*hy;
            //Если элемент находится на границе слоя, то не пересчитываем его
            if (i==0 || i==Nx-1 || j==0 || j==Ny-1) { //если боковые
                tmp_values[z*Nx*Ny+i*Nx+j] = values[z*Nx*Ny+i*Nx+j];
                continue;
            }
            int index = z*Nx*Ny+i*Nx+j;
            double tmp = (values[z*Nx*Ny+(i+1)*Nx+j]+values[z*Nx*Ny+(i-1)*Nx+j])/(hx*hx);
            tmp += (values[z*Nx*Ny+i*Nx+(j+1)]+values[z*Nx*Ny+i*Nx+(j-1)])/(hy*hy);
            tmp += (values[(z+1)*Nx*Ny+i*Nx+j]+values[(z-1)*Nx*Ny+i*Nx+j])/(hz*hz);
            tmp -= ro(cur_x, cur_y, cur_z);
            tmp_values[index] = 1/(2/(hx*hx)+2/(hy*hy)+2/(hz*hz)+a);
            tmp_values[index]*=tmp;
            max_delta = max(max_delta, fabs(tmp_values[index]-values[index]));
        }
    }
    return max_delta;
}

int main(int argc, char** argv) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (Nz % size) {  //нужно чтобы по z чётко делилось
        if (rank == 0) {
            cout << "Invalid number of processes" << endl;
        }
        MPI_Finalize();
        return 0;
    }

    double start = MPI_Wtime();

    double hx = (max_x - min_x) / (Nx - 1); //шаги сетки
    double hy = (max_y - min_y) / (Ny - 1);
    double hz = (max_z - min_z) / (Nz - 1);

    int part_z = Nz / size; //часть по z в процессе

    double* values = new double[(part_z + 2) * Nx * Ny];
    double* tmp_values = new double[(part_z + 2) * Nx * Ny];

    int base_z = rank * part_z-1;

    //Инициализация слоя
    for (int i = 0; i < part_z+2; i++) {
        int real_i = i + base_z;
        double cur_z = min_z + hz*real_i;
        for (int j = 0; j < Nx; j++) {
            double cur_x = min_x + hx*j;
            for (int k = 0; k < Ny; k++) {
                double cur_y = min_y + hy*k;
                int index = i*Nx*Ny + j*Nx + k;

                if (real_i == 0 || real_i == Nz-1 || j == 0 || j == Nx-1 || k == 0 || k == Ny-1) {
                    values[index] = phi_finction(cur_x, cur_y, cur_z);
                } else {
                    values[index] = 0;
                }
            }
        }
    }


    while(true) {
        double max_delta = 0;

        double tmp_delta = update_layer(base_z, 1, values, tmp_values, hx, hy, hz); //верх
        max_delta = max(max_delta, tmp_delta);

        tmp_delta = update_layer(base_z, part_z, values, tmp_values, hx, hy, hz); //низ
        max_delta = max(max_delta, tmp_delta);

        MPI_Request rq[4]; //хранят статус

        if (rank != 0) {
            MPI_Isend(tmp_values+Nx*Ny, Nx*Ny, MPI_DOUBLE, rank-1, 123, MPI_COMM_WORLD, &rq[0]);  //отправка + статус
            MPI_Irecv(tmp_values, Nx*Ny, MPI_DOUBLE, rank-1, 123, MPI_COMM_WORLD, &rq[2]);
        }

        if (rank != size-1) {
            MPI_Isend(tmp_values+part_z*Nx*Ny, Nx*Ny, MPI_DOUBLE, rank+1, 123, MPI_COMM_WORLD, &rq[1]);
            MPI_Irecv(tmp_values+(part_z+1)*Nx*Ny, Nx*Ny, MPI_DOUBLE, rank+1, 123, MPI_COMM_WORLD, &rq[3]);
        }

        for (int i = 2; i < part_z; i++) {  //каждый слой внутри
            double tmp_delta = update_layer(base_z, i, values, tmp_values, hx, hy, hz);
            max_delta = max(max_delta, tmp_delta);
        }

        if (rank != 0) {  //точно должны получить
            MPI_Wait(&rq[0], MPI_STATUS_IGNORE);
            MPI_Wait(&rq[2], MPI_STATUS_IGNORE);
        }

        if (rank != size-1) {
            MPI_Wait(&rq[1], MPI_STATUS_IGNORE);
            MPI_Wait(&rq[3], MPI_STATUS_IGNORE);
        }

        memcpy(values, tmp_values, (part_z+2)*Nx*Ny*sizeof(double));  //скопировали в основное
        double max_delta_shared;

        MPI_Reduce(&max_delta, &max_delta_shared, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); //выбираем макс

        MPI_Bcast(&max_delta_shared, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //передает данные от одного участника группы всем членам группы

        if (max_delta_shared < eps) {
            break;
        }
    }

    double* results;
    if (rank == 0) {
        results = new double[Nx*Ny*Nz];
    }

    MPI_Gather(values+Nx*Ny, part_z*Nx*Ny, MPI_DOUBLE, results, part_z*Nx*Ny, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Собирает данные от всех участников группы к одному участнику.

    if (rank == 0) { //грубо говоря проверка
        double max_delta = 0;
        for (int layer = 0;layer < Nz;layer++){
            double z = min_z + layer*hz;
            for (int j = 0;j < Nx;j++) {
                double x= min_x + j*hx;
                for (int k = 0; k < Ny; k++) {
                    double y = min_y + k*hy;
                    double tmp = results[layer*Nx*Ny + j*Nx + k];
                    double val = phi_finction(x, y, z);
                    max_delta = max(max_delta, fabs(tmp-val));
                }
            }
        }


        cout << "Delta: " << max_delta << endl;
        if(max_delta < 100*eps){
            cout << "Good" << endl;
        }
        else{
            cout << "Bad" << endl;
        }
        delete[] results;
    }

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Time: %lf\n", end - start);
    }

    delete[](values);
    delete[](tmp_values);

    MPI_Finalize();
    return 0;
}
