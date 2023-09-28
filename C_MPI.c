#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>


int main(int argc, char *argv[])
{
    int rank, numprocs, j, i;
    double *arrA, *arrB, *arrC, *arrD, *arrA_part, *arrB_part, *arrC_part, *arrD_part;
    double *temp_array_send, *temp_array_recv;
    double *A_extended, *A_extended_a, *A_extended_b, *A_extended_c, *A_extended_d;
    double *x_temp, *x_part_last, *x_part, *x;
    
    struct timeval etStart, etStop;
    struct timezone dummyTz;
    unsigned long long startTime, endTime;
    
    FILE* file = NULL;
    
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*a b c d - векторы одинаковой длины, описывающие систему(a1 = 0, сN = 0) - для разработки длина = 9*/
    int N = 9;

    /*каждому процессу достается 4 вектора меньшей длины - для трех процессов длины векторов = 3*/
    int N_part = N / numprocs;
    
    if (rank == 0)
    {
     arrA = (double*)calloc(N, sizeof(double));
     for(j = 0; j < N; j++) arrA[j] = 1;
     arrB = (double*)calloc(N, sizeof(double));
     for(j = 0; j < N; j++) arrB[j] = 2;
     arrC = (double*)calloc(N, sizeof(double));
     for(j = 0; j < N; j++) arrC[j] = 1;
     arrD = (double*)calloc(N, sizeof(double));
     for(j = 0; j < N; j++) arrD[j] = 1;
    }

    arrA_part = (double*)calloc(N_part, sizeof(double));
    arrB_part = (double*)calloc(N_part, sizeof(double));
    arrC_part = (double*)calloc(N_part, sizeof(double));
    arrD_part = (double*)calloc(N_part, sizeof(double));

    gettimeofday(&etStart, &dummyTz);
    MPI_Scatter(arrA, N_part, MPI_DOUBLE, arrA_part, N_part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(arrB, N_part, MPI_DOUBLE, arrB_part, N_part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(arrC, N_part, MPI_DOUBLE, arrC_part, N_part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(arrD, N_part, MPI_DOUBLE, arrD_part, N_part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank > 0 || rank == 0)
        {
        
            for (int n = 1; n < N_part; n++) {
                double coef = arrA_part[n]/arrB_part[n-1];
                arrA_part[n] = -coef*arrA_part[n-1];
                arrB_part[n] = arrB_part[n] - coef*arrC_part[n-1];
                arrD_part[n] = arrD_part[n] - coef*arrD_part[n-1];
            }

            for (int n = N_part-3; n > -1; n--) {
                double coef = arrC_part[n]/arrB_part[n+1];
                arrC_part[n] = -coef*arrC_part[n+1];
                arrA_part[n] = arrA_part[n] - coef*arrA_part[n+1];
                arrD_part[n] = arrD_part[n] - coef*arrD_part[n+1];
            }
        }

        //все процессы кроме 0-го будут отправлять данные
        if (rank > 0)
        {
           temp_array_send = (double*)calloc(4, sizeof(double));
           temp_array_send[0] = arrA_part[0];
           temp_array_send[1] = arrB_part[0];
           temp_array_send[2] = arrC_part[0];
           temp_array_send[3] = arrD_part[0];
        }

        //все процессы кроме последнего будут принимать данные
        if (rank < numprocs - 1)
        {
           temp_array_recv = (double*)calloc(4, sizeof(double));
        }

        //0-й процесс только принимает
        if (rank == 0)
        {
           MPI_Recv(temp_array_recv, 4, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        }

        //все процессы кроме нулевого и отправляют и получают сообщения
        if (rank > 0 && rank < numprocs-1)
        {
           MPI_Sendrecv(temp_array_send, 4, MPI_DOUBLE, rank-1, 0,
                        temp_array_recv, 4, MPI_DOUBLE, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }

        //последниц процесс только отправляет
        if (rank == numprocs-1)
        {
           MPI_Send(temp_array_send, 4, MPI_DOUBLE, numprocs-2, 0, MPI_COMM_WORLD);
        }

        if (rank < numprocs-1)
        {
           double coef = arrC_part[N_part-1]/temp_array_recv[1];
           arrB_part[N_part-1] = arrB_part[N_part-1] - coef*temp_array_recv[0];
           arrC_part[N_part-1] = -coef*temp_array_recv[2];
           arrD_part[N_part-1] = arrD_part[N_part-1] - coef*temp_array_recv[3];
        }
        
    //Массив на отправку: с каждого процесса последние элементы матрицы
        if (rank > 0 || rank == 0)
        {
           temp_array_send = (double*)calloc(4, sizeof(double));
           temp_array_send[0] = arrA_part[N_part-1];
           temp_array_send[1] = arrB_part[N_part-1];
           temp_array_send[2] = arrC_part[N_part-1];
           temp_array_send[3] = arrD_part[N_part-1];
        }
    
    //На нулевом процессе подготавливается рсширенная матрица системы
        if (rank == 0)
        {
         A_extended = (double*)calloc(numprocs*4, sizeof(double));
        }
        
    //Собираем столбцы в расширенную матрицу системы
        MPI_Gather(temp_array_send, 4, MPI_DOUBLE, A_extended, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
        x_temp = (double*)calloc(numprocs, sizeof(double));
        //double x_temp[numprocs];
        A_extended_a = (double*)calloc(numprocs, sizeof(double));
        A_extended_b = (double*)calloc(numprocs, sizeof(double));
        A_extended_c = (double*)calloc(numprocs, sizeof(double));
        A_extended_d = (double*)calloc(numprocs, sizeof(double));
    
        if (rank == 0)
        {
         i = 0;
         for(j = 0; j < numprocs*4; j = j+4){
            A_extended_a[i] = A_extended[j];
            i++;
         }
         i = 0;
         for(j = 1; j < numprocs*4; j = j+4){
            A_extended_b[i] = A_extended[j];
            i++;
         }
         i = 0;
         for(j = 2; j < numprocs*4; j = j+4){
            A_extended_c[i] = A_extended[j];
            i++;
         }
         i = 0;
         for(j = 3; j < numprocs*4; j = j+4){
            A_extended_d[i] = A_extended[j];
            i++;
         }
        }
        
    //Прямой ход метода Гуасса для нахождения вектора x_temp (размер которого = количеству процессов)
        if (rank == 0)
        {
         for (int i = 0; i < numprocs; i++){
             x_temp[i] = 0;  
         }
        //Обнуляем все элементы, лежащие ниже главной диагонали - прямой метод Гаусса
         for (int n = 1; n < numprocs; n++){
             double coef = A_extended_a[n]/A_extended_b[n-1];
             A_extended_b[n] = A_extended_b[n] - coef*A_extended_c[n-1];
             A_extended_d[n] = A_extended_d[n] - coef*A_extended_d[n-1]; 
         }

        //Обнуляем все элементы, лежащие выше главной диагонали - обратный ход метода Гаусса
         for (int n = numprocs-2; n > -1; n--){
            double coef = A_extended_c[n]/A_extended_b[n+1];
            A_extended_d[n] = A_extended_d[n] - coef*A_extended_d[n+1]; 
         }

        //Нахождение решения
         for (int n = 0; n < numprocs; n++){
            x_temp[n] = A_extended_d[n]/A_extended_b[n];
         }
        }
    
       int rcounts_temp[numprocs];
       int displs_temp[numprocs];
    //Подготовка массивов для работы со сдвигом в функции Scatter
       if (rank == 0) {
        rcounts_temp[0] = 1;
        displs_temp[0] = 0;
        for (int k = 1; k < numprocs; k++){
            rcounts_temp[k] = 2;
            displs_temp[k] = k - 1;
        }
       }
    
    //Записываем в x_part_last нужное количество элементов для каждого процесса из x_temp
       if (rank == 0) {
        x_part_last = (double*)calloc(1, sizeof(double));
        MPI_Scatterv(x_temp, rcounts_temp, displs_temp, MPI_DOUBLE,
                     x_part_last, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       }
       else{
        x_part_last = (double*)calloc(2, sizeof(double));
        MPI_Scatterv(x_temp, rcounts_temp, displs_temp, MPI_DOUBLE,
                     x_part_last, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       }
    
    //Создание конечных кусков массива x на каждом процессе
       x_part = (double*)calloc(N_part, sizeof(double));
    
       if (rank == 0) {
        if (N <= 20)
        printf("Для процесса № %d x_part = [ ", rank);
        for (int n = 0; n < N_part-1; n++){
            x_part[n] = (arrD_part[n] - arrC_part[n]*x_part_last[0])/arrB_part[n];
            x_part[N_part-1] = x_part_last[0];
            if (N <= 20)
            printf("%0.4f; ", x_part[n]);
        }
        if (N <= 20)
        printf("%0.4f ]\n\n", x_part[N_part-1]);
       }
       else {
        if (N <= 20)
        printf("Для процесса № %d x_part = [ ", rank);
        for (int n = 0; n < N_part-1; n++){
            x_part[n] = (arrD_part[n] - arrA_part[n]*x_part_last[0] - arrC_part[n]*x_part_last[1])/arrB_part[n];
            x_part[N_part-1] = x_part_last[1];
            
            if (N <= 20)
            printf("%0.4f; ", x_part[n]);
        }
        if (N <= 20)
        printf("%0.4f ]\n\n", x_part[N_part-1]);
       }
    
    //Собираем все части вектора решений на 0-м процессе
    x = (double*)calloc(N, sizeof(double));
    
    MPI_Gather(x_part, N_part, MPI_DOUBLE, x, N_part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gettimeofday(&etStop, &dummyTz);
    
    if (rank == 0) {
       if(N < 20){
       printf("\nМассив решений x = [ ");
       
            printf("%0.4f", x[0]);
            for (int n = 1; n < N; n++)
            {
              printf("; %0.4f", x[n]);
            }
            
            printf(" ]\n\n");
       }
       //Запись в файл
       file = fopen("C_MPI_val.txt", "w+b");
       if (file == NULL) {
           printf("Error in opening file... \n");
           exit(-1);
       }
       
       fprintf(file, "%0.4f", x[0]);
       for (int n = 1; n < N; n++) {
           fprintf(file, "\n%0.4f", x[n]);
       }
       fclose(file);
            
       startTime = (unsigned long long)etStart.tv_sec * 1000000 + etStart.tv_usec;
       endTime = (unsigned long long)etStop.tv_sec * 1000000 + etStop.tv_usec;
       /* Выводим время  работы параллельной части*/
       printf("\nTime = %g s.\n",(float)(endTime - startTime)/(float)1000000);
    }
    
    MPI_Finalize();
    
    return 0;
}
