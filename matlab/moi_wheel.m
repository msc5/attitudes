function moi = moi_wheel(m, h, r)
% Gives the moment of inertia of a solid wheel
% Parameters:
% m = mass                  (kg)
% h = height / thickness    (m)
% r = radius                (m)
a = (1/4) * m * r^2 + (1/12) * m * h^2;
b = (1/2) * m * r^2;
moi = diag([a, a, b]);