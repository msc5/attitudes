%% Imports

vect = vector;
quat = quaternion;
wm = wmap;

%% Simulate

% Simulation Parameters
T = 60 * 60;
n = 20000;
dt = T / n;
times = linspace(0, T, n);

q_errors = zeros(4, n);
orientation = zeros(4, n);
slide_norm = zeros(n);
w_errors = zeros(3, n);
wheel_momenta = zeros(3, n);
Q_C = zeros(4, n);

% Initial Conditions
J = [399, -2.81, -1.31; -2.81, 377, 2.54; -1.31, 2.54, 377];
J_hat = [380, -2.81, -1.31; -2.81, 360, 2.54; -1.31, 2.54, 340];
d = [0.005 * sin(0.05 * times); 0.003 * ones(n); 0.005 * cos(0.05 * times)];
h_0 = [0, 0, 0]';
w_0 = [0, 0, 0]';
Phi = 60 * (pi / 180);
eps = 0.01;

% Controller Parameters
phi = 0;
theta = 0.3927;
psi = 0;
phi_dot = 0.001745;
psi_dot = 0.04859;

k_p = 10;
k_d = 150;
k = 0.015;
G = 0.15 * eye(3);

% Generate Commands
X_C = zeros(n);
Y_C = zeros(n);
euler_angles = zeros(n, 3);
Q_C = zeros(n, 4);
W_C = zeros(n, 3);
W_C_dot = zeros(n, 3);
for i = 1:n
    [x, y] = wm.pos(phi, theta, psi);
    [q_c, w_c, w_c_dot] = wm.commands(phi, theta, psi);
    Q_C(i, :) = q_c;
    W_C(i, :) = w_c;
    W_C_dot(i, :) = w_c_dot;
    euler_angles(i, :) = [phi, theta, psi];
    X_C(i) = x;
    Y_C(i) = y;
    phi = phi + phi_dot * dt;
    psi = psi + psi_dot * dt;
end

%%

% Reaction Wheel Simulation
q = wm.q_0(Phi, Q_C(1, :)');
h = h_0;
w = w_0;
for i = 1:n
    
    q_c = Q_C(i, :)';
    w_c = W_C(i, :)';
    
    dq = quat.dq(q, q_c);

    s = (w - w_c) + k * sign(dq(4)) * dq(1:3);                      % 7.23a
    L = J * ((k / 2) * ((abs(dq(4)) * (w_c - w) - ...
        sign(dq(4)) * vect.cross(dq(1:3)) * (w + w_c))) + ...
        w_c_dot - G * vect.sat(s, eps)) + vect.cross(w) * J * w;
    
    q_dot = (1/2) * quat.xi(q) * w;
    w_dot = - inv(J) * (vect.cross(w) * J * w - L);
    h_dot = - vect.cross(w) * h - L;
    
    q = q + q_dot * dt;
    w = w + w_dot * dt;
    h = h + h_dot * dt;
    
    q_errors(:, i) = dq;
    orientation(:, i) = q;
    slide_norm(i) = norm(s);
    w_errors(:, i) = w - w_c;
    wheel_momenta(:, i) = h;
    
end


%% Plot Orientation

figure();
plot(times, orientation');
grid on;
yline(0,'k--');
yline(1,'k--');
ylabel('Quaternion');
xlabel('Time (s)');
legend('q_1', 'q_2', 'q_3', 'q_4', 'location', 'best');

%% Plot Total Wheel Momenta

% Wheel Momenta
figure();
plot(times, wheel_momenta);
grid on;
yline(0,'k--');
ylabel('Wheel Momenta (Nms)');
xlabel('Time (s)');
legend('h_1', 'h_2', 'h_3', 'location', 'best');

%%

% % Slide Norm
% figure();
% plot(times, slide_norm);
% grid on;
% yline(0,'k--');
% ylabel('Slide Norm');
% xlabel('Time (s)');


%%

% Errors
figure();
plot(times, w_errors');
grid on;
yline(0,'k--');
yline(1,'k--');
ylabel('Angular Rotation Errors');
xlabel('Time (s)');
legend('\delta \omega_1', '\delta \omega_2', '\delta \omega_3', '\delta \omega_4', 'location', 'best');