from mpi4py import MPI
from numpy import empty, array, int32, float64, dot
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()


N = 12
N_part = int(N / numprocs)

if rank == 0 :
     arrA = empty(N, dtype=float64)
     for j in range(0, N) : arrA[j] = 1
     arrB = empty(N, dtype=float64)
     for j in range(0, N) : arrB[j] = 2
     arrC = empty(N, dtype=float64)
     for j in range(0, N) : arrC[j] = 1
     arrD = empty(N, dtype=float64)
     for j in range(0, N) : arrD[j] = 1
else :
     arrA = None
     arrB = None
     arrC = None
     arrD = None

a_part = empty(N_part, dtype=float64);
b_part = empty(N_part, dtype=float64);
c_part = empty(N_part, dtype=float64);
d_part = empty(N_part, dtype=float64);

start = time.time()
comm.Scatter([arrA, N_part, MPI.DOUBLE], [a_part, N_part, MPI.DOUBLE], root=0)
comm.Scatter([arrB, N_part, MPI.DOUBLE], [b_part, N_part, MPI.DOUBLE], root=0)
comm.Scatter([arrC, N_part, MPI.DOUBLE], [c_part, N_part, MPI.DOUBLE], root=0)
comm.Scatter([arrD, N_part, MPI.DOUBLE], [d_part, N_part, MPI.DOUBLE], root=0)


for n in range(1, N_part) :
      coef = a_part[n]/b_part[n-1]
      a_part[n] = -coef*a_part[n-1]
      b_part[n] = b_part[n] - coef*c_part[n-1]
      d_part[n] = d_part[n] - coef*d_part[n-1]

for n in range(N_part -3, -1, -1):
        coef = c_part[n]/b_part[n+1]
        c_part[n] = -coef*c_part[n+1]
        a_part[n] = a_part[n] - coef*a_part[n+1]
        d_part[n] = d_part[n] - coef*d_part[n+1]

if rank > 0 :
        temp_array_send = array([a_part[0], b_part[0],
        c_part[0], d_part[0]],dtype=float64)
if rank < numprocs -1 :
        temp_array_recv = empty(4, dtype=float64)
if rank == 0 :
        comm.Recv([temp_array_recv , 4, MPI.DOUBLE], source=1,
        tag=0, status=None)
if rank in range(1, numprocs -1) :
        comm.Sendrecv(sendbuf=[temp_array_send , 4, MPI.DOUBLE],
        dest=rank -1, sendtag=0,
        recvbuf=[temp_array_recv , 4, MPI.DOUBLE],
        source=rank+1, recvtag=MPI.ANY_TAG , status=None)
if rank == numprocs -1 :
        comm.Send([temp_array_send , 4, MPI.DOUBLE],
        dest=numprocs -2, tag=0)
if rank < numprocs -1 :
        coef = c_part[N_part -1]/temp_array_recv[1]
        b_part[N_part -1] = b_part[N_part -1] - coef*temp_array_recv[0]
        c_part[N_part -1] = - coef*temp_array_recv[2]
        d_part[N_part -1] = d_part[N_part -1] - coef*temp_array_recv[3]

temp_array_send = array([a_part[N_part -1], b_part[N_part -1], c_part[N_part -1], d_part[N_part -1]],dtype=float64)

if rank == 0:
        A_extended = empty(numprocs*4, dtype=float64)
else:
        A_extended = None

comm.Gather([temp_array_send , 4, MPI.DOUBLE], [A_extended , 4, MPI.DOUBLE], root=0)

x_temp = empty(numprocs, dtype=float64)
A_extended_a = empty(numprocs, dtype=float64)
A_extended_b = empty(numprocs, dtype=float64)
A_extended_c = empty(numprocs, dtype=float64)
A_extended_d = empty(numprocs, dtype=float64)

if rank == 0:
         i = 0
         for j in range(0, numprocs*4, 4) : A_extended_a[i] = A_extended[j]; i = i+1
         i = 0
         for j in range(1, numprocs*4, 4) : A_extended_b[i] = A_extended[j]; i = i+1
         i = 0
         for j in range(2, numprocs*4, 4) : A_extended_c[i] = A_extended[j]; i = i+1
         i = 0
         for j in range(3, numprocs*4, 4) : A_extended_d[i] = A_extended[j]; i = i+1

#Находим x_temp
if rank == 0:
         #Обнуляем все элементы, лежащие ниже главной диагонали - прямой метод Гаусса
         for n in range(1, numprocs):
             coef = A_extended_a[n]/A_extended_b[n-1]
             A_extended_b[n] = A_extended_b[n] - coef*A_extended_c[n-1]
             A_extended_d[n] = A_extended_d[n] - coef*A_extended_d[n-1]
         #Обнуляем все элементы, лежащие выше главной диагонали - обратный ход метода Гаусса
         for n in range(numprocs-2, -1, -1):
            coef = A_extended_c[n]/A_extended_b[n+1]
            A_extended_d[n] = A_extended_d[n] - coef*A_extended_d[n+1]
         #Нахождение решения
         for n in range(numprocs): x_temp[n] = A_extended_d[n]/A_extended_b[n]

else:
        x_temp = None
    
if rank == 0:
        rcounts_temp = empty(numprocs, dtype=int32)
        displs_temp = empty(numprocs, dtype=int32)
        rcounts_temp[0] = 1
        displs_temp[0] = 0
        for k in range(1, numprocs) :
            rcounts_temp[k] = 2
            displs_temp[k] = k - 1
else :
        rcounts_temp = None; displs_temp = None

if rank == 0 :
        x_part_last = empty(1, dtype=float64)
        comm.Scatterv([x_temp, rcounts_temp, displs_temp, MPI.DOUBLE], [x_part_last, 1, MPI.DOUBLE], root = 0)
else :
        x_part_last = empty(2, dtype=float64)
        comm.Scatterv([x_temp, rcounts_temp, displs_temp, MPI.DOUBLE], [x_part_last, 2, MPI.DOUBLE], root = 0)

x_part = empty(N_part , dtype=float64)

if rank == 0 :
        if (N < 20) :
         print('Для процесса № %d x_part = [ ' % rank, end = '');
        for n in range(N_part-1) :
            x_part[n] = (d_part[n] - c_part[n]*x_part_last[0])/b_part[n]
            x_part[N_part -1] = x_part_last[0]
            if (N < 20):
             print('%.4f; ' % x_part[n], end = '')
        if (N < 20):
         print('%.4f ]\n' % x_part[N_part-1])
else :
        if (N < 20):
         print('Для процесса № %d x_part = [ ' % rank, end = '');
        for n in range(N_part-1) :
            x_part[n] = (d_part[n] - a_part[n]*x_part_last[0] - c_part[n]*x_part_last[1])/b_part[n]
            x_part[N_part -1] = x_part_last[1]
            if (N < 20):
             print('%.4f; ' % x_part[n], end = '')
        if (N < 20):
         print('%.4f ]\n' % x_part[N_part-1])

#Собираем все части вектора решений на 0-м процессе
x = empty(N, dtype=float64)

comm.Gather([x_part, N_part, MPI.DOUBLE], [x, N_part, MPI.DOUBLE], root=0)
end = time.time()

if (rank == 0) :
       if (N < 20):
        print('\nМассив решений x = [ ', end = '')
        print('%.4f;' % x[0], end = '')
        for n in range(1, N-1) : print('%.4f;' % x[n], end = '')
        print('%.4f ]\n' % x[N-1])

       f = open('Python_MPI_val.txt', 'w')
       f.write(str(round(x[0], 4)))
       for n in range(1, N-1):
        f.write('\n' + str(round(x[n], 4)))
       f.close()
       #время выводится в секундах
       print('\nTime = ', round((end - start), 6), end = ' s.')