%% Imports

clc; clear;
vect = vector;
quat = quaternion;

%% Simulate

% Simulation Parameters
T = 20 * 60;
n = 10000;
dt = T / n;
times = linspace(0, T, n);

errors = zeros(4, n);
wheel_momenta = zeros(3, n);

% Initial Conditions
J = [6400, -76.4, -25.6; -76.4, 4730, -40; -25.6, -40, 8160];
h_0 = [0, 0, 0]';
w_0 = [0.01, 0.01, 0.01]';
q_0 = (sqrt(2) / 2) * [1, 0, 0, 1]';
q_c = [0, 0, 0, 1]';

% Controller Parameters
k_p = 10;
k_d = 150;

% Reaction Wheel Simulation
h = h_0;
q = q_0;
w = w_0;
fprintf('Initial:   %.3f\n', norm(J * w_0));
for t = 1:n
   
    dq = quat.dq(q, q_c);
    L = - k_p * sign(dq(4)) * dq(1:3) - k_d * w;
    
    w_dot = - inv(J) * (vect.cross(w) * J * w - L);
    h_dot = - vect.cross(w) * h - L;
    q_dot = (1/2) * quat.xi(q) * w;
    
    q = q + q_dot * dt;
    w = w + w_dot * dt;
    h = h + h_dot * dt;
    
    errors(:, t) = dq;
    wheel_momenta(:, t) = h;
    
end


%% Plot Total Wheel Momenta

% Wheel Momenta
figure();
plot(times, wheel_momenta);
grid on;
yline(0,'k--');
ylabel('Wheel Momenta (Nms)');
xlabel('Time (s)');
legend('h_1', 'h_2', 'h_3', 'location', 'best');

% Errors
figure();
plot(times, errors');
grid on;
yline(0,'k--');
yline(1,'k--');
ylabel('Quaternion Errors');
xlabel('Time (s)');
legend('q_1', 'q_2', 'q_3', 'q_4', 'location', 'best');

%% Plot Wheel Momenta Distributions

W_pyramid = (1 / sqrt(2)) * [1, -1, 0, 0; 1, 1, 1, 1; 0, 0, 1, -1];
W_NASA = [1, 0, 0, 1 / sqrt(3); 0, 1, 0, 1 / sqrt(3); 0, 0, 1, 1 / sqrt(3)];

H_wW_pyramid = pinv(W_pyramid) * wheel_momenta;
H_wW_NASA = pinv(W_NASA) * wheel_momenta;

% Pyramid Wheel Momenta
figure();
plot(times, H_wW_pyramid);
grid on;
yline(0,'k--');
ylabel('Wheel Momenta (Nms)');
xlabel('Time (s)');
legend('w_1', 'w_2', 'w_3', 'w_4', 'location', 'best');

% NASA Wheel Momenta
figure();
plot(times, H_wW_NASA);
grid on;
yline(0,'k--');
ylabel('Wheel Momenta (Nms)');
xlabel('Time (s)');
legend('w_1', 'w_2', 'w_3', 'w_4', 'location', 'best');

fprintf('Pyramid:   %.3f\n', norm(H_wW_pyramid(:, end)));
fprintf('NASA:      %.3f\n', norm(H_wW_NASA(:, end)));

