#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>
#include <pthread.h>

FILE* file = NULL;
int N = 9;
int numberOfProcessors, numberOfThreadsPerProcessor, totalNumberOfThreads;
volatile float A[1000][1000], b[1000], x[1000];

void initializeMatrix();
void printInputs();
void printX();
void parallelThreadP();
void parallelThreadPEnoughThreadsToProcessAllRows();
void parallelThreadPLesserThreadsToProcessAllRows();
void *rowFactorMultiplicationWithoutSkipLogic(struct ThreadParam * threadParam);
void *rowFactorMultiplicationWithSkipLogic(struct ThreadParam * threadParam);
void backSubstitution();
void gauss();

struct ThreadParam{
    int threadId;
    int outerRow;
};

int main(int argc, char **argv) {

    struct timeval etStart, etStop;
    struct timezone dummyTz;
    unsigned long long startTime, endTime;
    
    numberOfThreadsPerProcessor = atoi(argv[1]);
    numberOfProcessors = 1;

    initializeMatrix();
    printMatrix();
    gettimeofday(&etStart, &dummyTz);
    parallelThreadP();
    gettimeofday(&etStop, &dummyTz);
    printX();
    
    startTime = (unsigned long long)etStart.tv_sec * 1000000 + etStart.tv_usec;
    endTime = (unsigned long long)etStop.tv_sec * 1000000 + etStop.tv_usec;
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
       file = fopen("C_Pthreads_val.txt", "w+b");
       if (file == NULL) {
           printf("Error in opening file... \n");
           exit(-1);
       }
       
       fprintf(file, "%0.4f", x[0]);
       for (n = 1; n < N; n++) {
           fprintf(file, "\n%0.4f", x[n]);
       }
       fclose(file);
       printf("The values of x are wtitten in C_Pthreads_val.txt");
}

void parallelThreadP(){
    int norm, row, col;  
    float multiplier;
    totalNumberOfThreads = ((N-1)>(numberOfProcessors*numberOfThreadsPerProcessor))?(numberOfProcessors*numberOfThreadsPerProcessor):(N-1);
    if(totalNumberOfThreads == N-1){
       parallelThreadPEnoughThreadsToProcessAllRows(); 
    }else{
        parallelThreadPLesserThreadsToProcessAllRows(); 
    }
    backSubstitution();
}

void parallelThreadPEnoughThreadsToProcessAllRows(){
    pthread_t pthreads[totalNumberOfThreads];
    struct ThreadParam* param = malloc(totalNumberOfThreads* sizeof(struct ThreadParam));
    int outerRow;
    for(outerRow = 0 ; outerRow < N-1; outerRow++){
        int threadIndex;
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            param[threadIndex].threadId = threadIndex;
            param[threadIndex].outerRow = outerRow;
            pthread_create(&pthreads[threadIndex],NULL,rowFactorMultiplicationWithoutSkipLogic,&param[threadIndex]);
        }
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            pthread_join(pthreads[threadIndex],NULL);
        }
    }
    free(param);
}

void parallelThreadPLesserThreadsToProcessAllRows(){
    pthread_t pthreads[totalNumberOfThreads];
    struct ThreadParam *param = malloc(totalNumberOfThreads* sizeof(struct ThreadParam));

    int outerRow;
    for(outerRow = 0 ; outerRow < N-1; outerRow++){
        int threadIndex;
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            param[threadIndex].threadId = threadIndex;
            param[threadIndex].outerRow = outerRow;
            pthread_create(&pthreads[threadIndex],NULL,rowFactorMultiplicationWithSkipLogic,&param[threadIndex]);
        }
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            pthread_join(pthreads[threadIndex],NULL);
        }
    }
    free(param);
}

void *rowFactorMultiplicationWithoutSkipLogic(struct ThreadParam * threadParam){
    int col;
    float multiplier;
    int outerRow = threadParam->outerRow;
    int innerRow = threadParam->threadId+outerRow+1;
    if(innerRow < N) {
        multiplier = A[innerRow][outerRow] / A[outerRow][outerRow];
        for (col = outerRow; col < N; col++) {
            A[innerRow][col] -= A[outerRow][col] * multiplier;
        }
        b[innerRow] -= b[outerRow] * multiplier;
    }
}

void *rowFactorMultiplicationWithSkipLogic(struct ThreadParam * threadParam){
    int innerRow, col;
    float multiplier;
    int startIndex = threadParam->threadId+1;
    int outerRow = threadParam->outerRow;
    for (innerRow = startIndex+outerRow; innerRow < N; innerRow+=totalNumberOfThreads) {
        multiplier = A[innerRow][outerRow] / A[outerRow][outerRow];
        for (col = outerRow; col < N; col++) {
            A[innerRow][col] -= A[outerRow][col] * multiplier;
        }
        b[innerRow] -= b[outerRow] * multiplier;
    }
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
