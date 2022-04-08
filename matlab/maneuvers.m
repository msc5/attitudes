%% Imports

reg = regulate;
quat = quaternion;

%% Spacecraft 

% Spacecraft Moment of Inertia (wet) (kg m^2)
J = [4.7921880e+06  8.4185790e+03  1.5313532e+04;
     8.4185790e+03  9.3203071e+06 -1.2106130e+04;
     1.5313532e+04 -1.2106130e+04  1.1217469e+07] * 0.00142233;


%% Disturbance Torques (SRP, GG, etc.)

T = 40 * 60;                % Time to simulate (s)
SRP = 3.46e-5;              % SRP Torque (Nm)
w_0 = [0; 0; 0];            % Initial Angular Rotation (rad/s)
q_c = [0; 0; 0; 1];         % Command Quaternion

[times, errors, momenta, X] = reg.regulate(J, w_0, q_c, SRP, T);
reg.plot_momenta(times, errors, momenta);

[pyramid, nasa] = reg.decompose(momenta);
reg.plot_wheel_momenta(times, pyramid, nasa);

reg.plot_rotations(X);

%% Detumble (Thruster Misfire, Launch Vehicle, etc.)

T = 60 * 60;                % Time to simulate (s)
w_0 = [0.1; 0.5; 0];        % Initial Angular Rotation (rad/s)
q_c = [0; 0; 0; 1];         % Command Quaternion

[times, errors, momenta, X] = reg.regulate(J, w_0, q_c, 0, T);
f = reg.plot_momenta(times, errors, momenta);
saveas(f, 'Detumble_Momenta.png');

[pyramid, nasa] = reg.decompose(momenta);
f = reg.plot_wheel_momenta(times, pyramid, nasa);
saveas(f, 'Detumble_Wheel_Momenta.png');

f = reg.plot_rotations(X);
saveas(f, 'Detumble_Rotations.png');

%% Belly Flop (180 degree turn)

T = 40 * 60;                    % Time to simulate (s)
w_0 = [0; 0; 0];                % Initial Angular Rotation (rad/s)
q_c = quat.q([0; 0; 1], pi);    % Command Quaternion

[times, errors, momenta, X] = reg.regulate(J, w_0, q_c, 0, T);
reg.plot_momenta(times, errors, momenta);

[pyramid, nasa] = reg.decompose(momenta);
reg.plot_wheel_momenta(times, pyramid, nasa);

reg.plot_rotations(X);

