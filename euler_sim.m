dt = 0.01;
t = 0:dt:10;
n = length(t);

% I = moi_wheel(10, 0.2, 0.4);            % (mass: kg, height: m, radius: m)
I = diag([5; 6; 10]);

M = rand(n, 3) * 10;
w = [5; 10; 3];

results = zeros(n, 3);
for i = 1:n
    w_dot = euler_eqn(I, M(i), w);
    w = w + dt * w_dot;
    results(i, :) = w;
end

figure('position', [500, 200, 1000, 1000]);
plot(results)
legend('\omega_x', '\omega_y', '\omega_z', 'location', 'northwest')