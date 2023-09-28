n = 80;
e = ones(n,1);
A = spdiags([e 2*e e], -1:1, n, n);
full(A)
B=[1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1]; % правая часть матрицы
B = ones(80,1)
det(A) % проверка, что определитель не равен 0
% метод Гаусса
x = A\B;
x = round(x, 4);
writematrix(x, 'expected_val.txt');