
a = 100;
A = [0, 1, 0; 0, 0, 1; 0, -a, -a - 1];
B = [0; 0; a];
C = [1, 0, 0];

poles = [-2+2i, -2-2i, -4];
K = place(A, B, poles);

k_r = -1 / (C * inv(A - B * K) * B);

sys = ss(A - B * K, B * k_r, C, 0);

step(sys);