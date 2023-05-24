#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "mpi.h"

#define P1 2
#define P2 2


#define N1 1000
#define N2 1000
#define N3 1000


void SetSettingArrs(int C_x, int C_y, int C_part_x, int C_part_y, int size, int* displs, int* block_size)
{
    int id = 0;
    for (int i = 0; i < C_y / C_part_y; ++i)
    {
        for (int j = 0; j < C_x / C_part_x; ++j)
        {
            block_size[id] = 1;
            displs[id] = j + i * (C_x / C_part_x * C_part_y);
            id++;
        }
    }
}

void FillMatrix(int size_x, int size_y, double* A)
{
    for (int i = 0; i < size_y; ++i)
    {
        double* a = A + i * size_x;
        for (int j = 0; j < size_x; ++j)
        {
            //Generate random double number with 2 decimal places from [-100; 100] 
            double rand = double(std::rand() % 10000) / 100;
            int sign = std::rand() % 2;
            if (sign == 1)
            {
                rand *= -1;
            }
            a[j] = rand;
        }
    }
}

void InitMatrices(int A_x, int A_y, double* A, int B_x, int B_y, double* B)
{
    FillMatrix(A_x, A_y, A);
    FillMatrix(B_x, B_y, B);
}

void SendPartMatrixA(const int* coords, int send_count, int recv_count, double* A, double* A_part, MPI_Comm column_comm, MPI_Comm row_comm)
{
    if (coords[1] == 0) //For processes from 1st column 2D lattice
    {
        MPI_Scatter(A, send_count, MPI_DOUBLE, A_part, recv_count, MPI_DOUBLE, 0, column_comm); //Each column_comm has its own "root" == 0 process
                                                                                    //Распределяет данные от одного члена по всем членам групп
                                                                                    //Функция распределения блоков данных по всем процессам группы
                                                                                    // Рассылаем в ячейки слева для начала
    }

    MPI_Bcast(A_part, send_count, MPI_DOUBLE, 0, row_comm);//Each row_comm has its own "root" == 0 process
                                                       //Передает данные от одного участника группы всем членам группы.
}

void SendPartMatrixB(const int* coords, int B_x, int B_y, int B_part_x, int B_part_y, double* B, double* B_part, MPI_Comm column_comm, MPI_Comm row_comm)
{
    //Creating column type
    MPI_Datatype column, column_type;
    MPI_Type_vector(B_y, B_part_x, B_x, MPI_DOUBLE, &column); //b_y элементов в векторе, B_part_x в каждом блоке,  B_x шаг
    MPI_Type_commit(&column); //фиксирует Объект типа данных должен быть зафиксирован, прежде чем его можно будет использовать в обмене данными.

    MPI_Type_create_resized(column, 0, B_part_x * sizeof(double), &column_type); //преобразуем column в column_type
    //в общем случае приходится корректировать длину структурированного типа, чтобы учесть любое завершающее дополнение, которое компилятор может вставить в конец структуры
    //Это вляет на поведение типа данных при передаче
    
    MPI_Type_commit(&column_type);

    if (coords[0] == 0) //For processes from 1st row 2D lattice
    {
        MPI_Scatter(B, 1, column_type, B_part, B_part_x * B_part_y, MPI_DOUBLE, 0, row_comm); //Each row_comm has its own "root" == 0 process
    }
    MPI_Bcast(B_part, B_part_x * B_part_y, MPI_DOUBLE, 0, column_comm); //Each column_comm has its own "root" == 0 process

    MPI_Type_free(&column);
    MPI_Type_free(&column_type);
}

void MultMatrix(int a_x, int a_y, int b_x, double* A, double* B, double* C)
{
    for (int i = 0; i < a_y; ++i)
    {
        double* c = C + i * b_x;
        for (int l = 0; l < b_x; ++l)
        {
            c[l] = 0;
        }

        for (int j = 0; j < a_x; ++j)
        {
            double a = A[i * a_x + j];
            double* b = B + j * b_x;
            for (int k = 0; k < b_x; ++k)
            {
                c[k] += a * b[k];
            }
        }
    }
}

void MatrAssembly(int size, int C_x, int C_y, int C_part_x, int C_part_y, double* C_part, double* C, MPI_Comm comm2d)
{
    //Creating block type
    MPI_Datatype block, block_type;
    MPI_Type_vector(C_part_y, C_part_x, C_x, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Type_create_resized(block, 0, C_part_x * sizeof(double), &block_type);
    MPI_Type_commit(&block_type);

    //Set auxiliary arrays
    auto displs = new int[size];
    auto block_size = new int[size];
    SetSettingArrs(C_x, C_y, C_part_x, C_part_y, size, displs, block_size);

    //Matrix assembly
    MPI_Gatherv(C_part, C_part_x * C_part_y, MPI_DOUBLE, C, block_size, displs, block_type, 0, comm2d);
    MPI_Type_free(&block);
    MPI_Type_free(&block_type);
    delete[] displs;
    delete[] block_size;
}

int CheckAnsw(int A_x, int A_y, int B_x, double* A, double* B, double* C)
{
    auto Right_answ = new double[A_y * B_x];
    int misstakes = 0;
    MultMatrix(A_x, A_y, B_x, A, B, Right_answ);
    for (int i = 0; i < A_y; ++i)
    {
        double* c = C + i * B_x;
        double* answ = Right_answ + i * B_x;
        for (int j = 0; j < B_x; ++j)
        {
            if (answ[j] - c[j] != 0)
            {
                misstakes++;
            }
        }
    }
    delete[]Right_answ;
    return misstakes;
}

int main(int argc, char* argv[])
{

    int dims[2] = { P1, P2 }, periods[2] = { 0, 0 }; //размер решетки и зацикливание в поле сетки??

    int coords[2], varying_coords[2];
    int reorder = 0;
    int size, rank;

    double start, end;

    MPI_Comm comm2d;
    MPI_Comm row_comm;
    MPI_Comm column_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int A_part_x = N1;
    int A_part_y = N2 / P1;
    int B_part_x = N3 / P2;
    int B_part_y = N1;

    //Memory allocation for matrices
    auto A = new double[N1 * N2];
    auto B = new double[N3 * N1];
    auto C = new double[N2 * N3];

    auto A_part = new double[A_part_x * A_part_y];
    auto B_part = new double[B_part_x * B_part_y];
    auto C_part = new double[A_part_y * B_part_x];

    //Creating a communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d); //Функция создания коммуникатора с декартовой топологией
                                                                     //создаёт коммуникатор решётки так чтобы процессы в рамках решетки могли коммуницироать со своими соседями
    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_coords(comm2d, rank, 2, coords); //запись координат в зависимости от rank

    //Creating a subcommunicator for rows
    varying_coords[0] = 0;
    varying_coords[1] = 1;
    MPI_Cart_sub(comm2d, varying_coords, &row_comm);

    //Creating a subcommunicator for columns
    varying_coords[0] = 1;
    varying_coords[1] = 0;
    MPI_Cart_sub(comm2d, varying_coords, &column_comm);


    if (rank == 0)
    {
        InitMatrices(N1, N2, A, N3, N1, B);
    }

    start = MPI_Wtime();

    //Разрезаем
    SendPartMatrixA(coords, A_part_x * A_part_y, N1 * N2, A, A_part, column_comm, row_comm);
    SendPartMatrixB(coords, N3, N1, B_part_x, B_part_y, B, B_part, column_comm, row_comm);

    //Умножение
    MultMatrix(A_part_x, A_part_y, B_part_x, A_part, B_part, C_part);

    //Собираем в одну матрицу
    MatrAssembly(size, N3, N2, B_part_x, A_part_y, C_part, C, comm2d);
    end = MPI_Wtime();

    if (rank == 0)
    {
        std::cout << " Misstakes: " << CheckAnsw(N1, N2, N3, A, B, C) << "      " << std::endl;
        std::cout << " Times: " << std::fixed << std::setprecision(3) << end - start << " sec." << std::endl;
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] A_part;
    delete[] B_part;
    delete[] C_part;
    MPI_Finalize();
    return 0;
}
