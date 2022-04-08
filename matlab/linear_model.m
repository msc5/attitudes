%% Imports

lin = linear;
plot_conf = plot_config;

%% Plant

J = diag([10000, 9000, 12000]);
w = [0.01, 0.05, 5]';
% w = [0, 15, 20]';
% w = [0, 1, 0.5]';

[A, B, C] = lin.ss(J, w);

P = ss(A, B, C, 0);

%% Controller

Q = diag([1, 1, 1, 0, 0, 0]);
R = eye(3);

K = lqr(A, B, Q, R);
kr = - (C * inv(A - B * K) * B) \ eye(3);

%% System

sys = ss(A - B * K, B * kr, C, 0);

%% Simulation Parameters

T = 10;
n = 1000;
t = linspace(0, T, n);

%% Step Response

w_o = [w; zeros(3, 1)];
% opt = stepDataOptions;
% opt.StepAmplitude = w_o';

% s = step(sys, t, opt);
s = step(sys, t);
error = 1 - s;

fig = figure();
fig.Position = plot_conf.size;
hold on;
grid on;
yline(1, 'k--');
for i = 1:3
    plot(t, s(:, i, i));
end
title({'Step Response of Linearized', 'Satellite Attitude Control System'});
legend('', '\omega_1', '\omega_2', '\omega_3', 'location', 'best');
xlabel('Time (s)');
ylabel('Step Response');

% saveas(fig, 'img/step_response.jpg');

%% Error 

fig = figure();
fig.Position = plot_conf.size;
hold on;
grid on;
yline(0, 'k--');
for i = 1:3
    plot(t, error(:, i, i));
end
title({'Error to Step Response of Linearized', 'Satellite Attitude Control System'});
legend('', 'e_1', 'e_2', 'e_3', 'location', 'best');
xlabel('Time (s)');
ylabel('Error');

% saveas(fig, 'img/error_response.jpg');

%% Error Analysis

T = 100;
n = 1000;
t = linspace(0, T, n);
s = step(sys, t);
error = 1 - s;

error_a = 1e-4;         % Acceptable spin rate error magnitude
error_norm = vecnorm([error(:, 1, 1), error(:, 2, 2), error(:, 3, 3)]');
t_a = find(error_norm < error_a);
t_a = t(t_a(1));

% t_a is the amount of time it takes to reach an amount of spin rate error
% within an acceptable range.

% Since the system is linear, it is always possible to reach 0 steady state
% error, given an infinite amount of time. The other limiting factors for
% this are errors propagated by the attitude determination systems and
% physical attitude control systems, i.e., reaction wheels, thrusters, etc.
