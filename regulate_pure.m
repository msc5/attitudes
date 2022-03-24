
vect = vector;
quat = quaternion;

% Simulation Parameters
T = 300;
n = 10000;
dt = T / n;
times = linspace(0, T, n);

errors = zeros(4, n);
control_torques = zeros(3, n);

% Initial Conditions
J = diag([10000, 9000, 12000]);
w_0 = [0.5300, 0.5300, 0.0530]' * (pi / 180);
q_0 = [0.6853, 0.6953, 0.1531, 0.1531]';
q_c = [0, 0, 0, 1]';

% Controller Parameters
k_p = 50;
k_d = 500;


%% Pure Torque Simulation

q = q_0;
w = w_0;
for t = 1:n
    
    dq = quat.dq(q, q_c);
    L = - k_p * sign(dq(4)) * dq(1:3) - k_d * w;
    w_dot = - inv(J) * (vect.cross(w) * J * w - L);
    q_dot = (1/2) * quat.xi(q) * w;
    q = q + q_dot * dt;
    w = w + w_dot * dt;
    
    errors(:, t) = dq;
    control_torques(:, t) = L;
    
end


%%

% Control Torques
plot(times, control_torques');
yline(0,'k--');
legend('L_1', 'L_2', 'L_3', 'location', 'southeast');

%%

% Errors
plot(times, errors');
legend('q_1', 'q_2', 'q_3', 'q_4');