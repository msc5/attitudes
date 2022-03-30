function w_dot = euler_eqn(I, M, w)

% Solves Euler's Equation for Rigid Body Dynamics for w_dot
% (Euler's Equation Given by I * w_dot + w x (I * w) = M)

w_dot = I \ (cross(I * w, w) + M);
