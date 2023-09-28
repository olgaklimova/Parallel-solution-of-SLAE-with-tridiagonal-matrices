#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>


FILE* file = NULL;
int N = 9;
int numthreads;
volatile float A[1000][1000], b[1000], x[1000];
void initializeMatrix();
void printMatrix();
void parallelOpenMP();
void printX();
void backSubstitution();

int main(int argc, char **argv) {

    struct timeval etStart, etStop;  
    struct timezone dummyTz;
    unsigned long long startTime, endTime;
    
    numthreads = atoi(argv[1]);
    
    //создаем трехдиагональную матрицу
    initializeMatrix();
    //выводим матрицу, если N<20
    printMatrix();
    
    //время до параллельного алгоритма
    gettimeofday(&etStart, &dummyTz);
    parallelOpenMP();
    //время после параллельного алгоритма
    gettimeofday(&etStop, &dummyTz);
    
    //выводим полученные значения,если N<20 (иначе просто записываем в  С_OpenP_val.txt)
    printX();
    
    startTime = (unsigned long long)etStart.tv_sec * 1000000 + etStart.tv_usec;
    endTime = (unsigned long long)etStop.tv_sec * 1000000 + etStop.tv_usec;
    //выводим время в милисекундах
    printf("\nTime = %g ms.\n",(float)(endTime - startTime)/(float)1000);

    exit(0);

}

void initializeMatrix() {
    int row, col;

    for (col = 0; col < N; col++) {
        for (row = 0; row < N; row++) {
            if (col == row-1)
               A[row][col] = 1;
            if (col == row)
               A[row][col] = 2;
            if (col == row+1)
               A[row][col] = 1;
            if (col != row-1 && col != row && col != row+1)
               A[row][col] = 0;
        }
        b[col] = 1.0;
        x[col] = 0.0;
    }
}

void printMatrix() {
    int row, col;

    if (N < 20) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%0.4f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
        printf("\nb = [");
        for (col = 0; col < N; col++) {
            printf("%0.4f%s", b[col], (col < N-1) ? "; " : "]\n");
        }
    }
}

void printX() {
    int n;

    if (N < 20) {
        printf("\nx = [");
        for (n = 0; n < N; n++) {
            printf("%0.4f%s", x[n], (n < N-1) ? "; " : "]\n");
        }
    }

    //Запись в файл
       file = fopen("C_OpenMP_val.txt", "w+b");
       if (file == NULL) {
           printf("Error in opening file... \n");
           exit(-1);
       }
       fprintf(file, "%0.4f", x[0]);
       for (n = 1; n < N; n++) {
           fprintf(file, "\n%0.4f", x[n]);
       }
       fclose(file);
       printf("The values of x are wtitten in C_OpenMP_val.txt");
}

void backSubstitution(){
    int norm, row, col;

    for (row = N - 1; row >= 0; row--) {
        x[row] = b[row];
        for (col = N-1; col > row; col--) {
            x[row] -= A[row][col] * x[col];
        }
        x[row] /= A[row][row];
    }
}

void parallelOpenMP() {
    int norm, row, col;
    float multiplier;

    omp_set_num_threads(numthreads);
    for (norm = 0; norm < N - 1; norm++) {
    #pragma omp parallel for shared(A,b,norm,N) private(row,multiplier,col) default(none)
        for (row = norm + 1; row < N; row++) {
          /*int tid = omp_get_thread_num();
            printf("Hy from thread = %d\n", tid);*/
            multiplier = A[row][norm] / A[norm][norm];
            for (col = norm; col < N; col++) {
                A[row][col] -= A[norm][col] * multiplier;
            }
            b[row] -= b[norm] * multiplier;
        }
    }

    backSubstitution();
}
