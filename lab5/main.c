#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int iterCounter = 4;
int L = 500;
int *task_list;
int nextPosition;

double globalRes = 0;
int tasksCount = 28000;
int processTasksCount;
int iterTaskCount;

int size;
int rank;

pthread_t threads[2];
pthread_mutex_t mutex;



void initList(int *taskList, int procTaskCount, int iterCount){ //вес задач инициализации
    for(int i = 0; i < procTaskCount; i++){
        taskList[i] = abs(50 - i%100)*abs(rank - (iterCount % size))*L;
    }
}

void* doTasks(void* args){

    int currListNum = 0;

    double start_iteration;
    double end_iteration;

    double time_m;
    double time_n;

    MPI_Status st;

    int request;

    int iterationCompletedTasksNum;

    while (currListNum != iterCounter){

        initList(task_list, processTasksCount, currListNum);
        iterationCompletedTasksNum = 0;
        nextPosition = 0;
        start_iteration = MPI_Wtime();
        iterTaskCount = processTasksCount;

        while(iterTaskCount != 0){
            pthread_mutex_lock(&mutex); //блокирование очереди задач
            int weight = task_list[nextPosition]; //вес текущ задачи
            nextPosition++;
            iterTaskCount--;
            pthread_mutex_unlock(&mutex); //разблокировка очереди задач

            for(int i = 0; i < weight; i++){
                globalRes += sin(i);
            }

            iterationCompletedTasksNum++;
        }


        for(int i = 0; i < size; i++){
            request = 1;

            if((rank + i) % size != rank){
                while(1){

                    MPI_Send(&request, 1, MPI_INT, (rank + i) % size, 0, MPI_COMM_WORLD);

                    int response;


                    MPI_Recv(&response, 1, MPI_INT, (rank + i) % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                    if(response != -1){
                        int *response_tasks = (int*)malloc(sizeof(int)*response);
                        MPI_Recv(response_tasks, response, MPI_INT, (rank + i) % size, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        for(int j = 0; j < response; j++){
                            for(int k = 0; k < response_tasks[j]; k++){
                                globalRes += sin(k);
                            }
                            iterationCompletedTasksNum++;
                        }

                        free(response_tasks);
                    }
                    else break;
                }
            }
        }

        end_iteration = MPI_Wtime();
        double iterationTimeProc =  end_iteration - start_iteration;
        printf("Process#%d | TasksIterationCount:%d | IterationTime:%f\n", rank, iterationCompletedTasksNum, iterationTimeProc);

        MPI_Allreduce(&iterationTimeProc, &time_m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Объединяет значения из всех процессов и распределяет результат обратно во все процессы.
        MPI_Allreduce(&iterationTimeProc, &time_n, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); // Ищем мин и макс время

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0){
            printf("Imbalance:%f\n", time_m - time_n);
            printf("Share of imbalance:%.2f\n", ((time_m - time_n)/time_m) * 100);
        }

        double globalResIteration;
        MPI_Allreduce(&globalRes, &globalResIteration, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(rank == 0)
            printf("GlobalRes iteration:%.3f\n", globalResIteration);

        currListNum++;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    request = 0;
    MPI_Send(&request, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);

    return NULL;
}


void* sendTask(void* args){
    MPI_Status status;
    int request;
    int response;

    while(1){

        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        if(request == 0)break;

        pthread_mutex_lock(&mutex);
        int* sendTasks;

        if(iterTaskCount > 100){
            response = 50;

            sendTasks = (int*)malloc(sizeof(int)*response);
            for(int i = 0; i < response; i++){
                sendTasks[i] = task_list[nextPosition];
                nextPosition++;
                iterTaskCount--;
            }
        }
        else {
            response = -1;
        }

        pthread_mutex_unlock(&mutex);

        MPI_Send(&response, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);

        if(response > 0){
            MPI_Send(sendTasks, response, MPI_INT, status.MPI_SOURCE, 2, MPI_COMM_WORLD);
            free(sendTasks);
        }

    }

    return NULL;
}

void createThreads(){

    pthread_attr_t attr;

    pthread_attr_init(&attr); //получаем дефолтные значения атрибутов

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_create(&threads[0], &attr, doTasks, NULL); // создание потока получения
    pthread_create(&threads[1], &attr, sendTask, NULL);

    pthread_attr_destroy(&attr);  //уничтожение атрибутов

    pthread_join(threads[0], NULL); //ждем пока другой поток завершится
    pthread_join(threads[1], NULL);
}

int main(int argc, char** argv) {

    int provided;
    double start;
    double end;
    double generalTime;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); //Инициализирует среду выполнения вызывающего процесса

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(provided != MPI_THREAD_MULTIPLE){
        printf("Cannot get needed provided level!\n");
    }

    pthread_mutex_init(&mutex, NULL);
    processTasksCount = tasksCount / size;

    if (rank < tasksCount % size) {
        processTasksCount = tasksCount / size + 1;
    }
    else {
        processTasksCount = tasksCount / size;
    }

    task_list = (int*)malloc(processTasksCount * sizeof(int));

    start = MPI_Wtime();

    createThreads();
    end = MPI_Wtime();

    double time = end - start;

    MPI_Reduce(&time, &generalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Time:%f\n", generalTime);
    }

    pthread_mutex_destroy(&mutex);
    free(task_list);
    MPI_Finalize();

    return 0;
}
