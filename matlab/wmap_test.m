
wm = wmap;

%%

% Simulation Parameters
T = 1 * 60 * 60;
n = 1000;
dt = T / n;
times = linspace(0, T, n);

phi_dot = 0.001745;
psi_dot = 0.04859;

phi = 0;
theta = 0.3927;
psi = 0;

X = zeros(n);
Y = zeros(n);
Angles = zeros(n, 3);
Q = zeros(n, 4);
W = zeros(n, 3);
W_dot = zeros(n, 3);
for i = 1:n
    [x, y] = wm.pos(phi, theta, psi);
    [q_c, w_c, w_c_dot] = wm.commands(phi, theta, psi);
    Q(i, :) = q_c;
    W(i, :) = w_c;
    W_dot(i, :) = w_c_dot;
    Angles(i, :) = [phi, theta, psi];
    X(i) = x;
    Y(i) = y;
    phi = phi + phi_dot * dt;
    psi = psi + psi_dot * dt;
end

%%

plot(X, Y);

%%

plot(Angles);

%%

plot(W);

%%

plot(Q);