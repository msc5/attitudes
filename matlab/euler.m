function [times, momenta, X] = euler(J, w_0, M, T)

    % Simulation Parameters
    n = 10000;
    dt = T / n;
    times = linspace(0, T, n);
    
    % Initial Basis for display
    X = zeros(n, 3, 3);
    x_0 = eye(3);
    momenta = zeros(n, 3);
    
    h = h_0;
    q = q_0;
    q_c = q_0;
    w = w_0;
    for t = 1:n

        w_dot = - inv(J) * (vect.cross(w) * J * w - M);
        h_dot = - vect.cross(w) * h;
        q_dot = (1/2) * quat.xi(q) * w;

        q = q + q_dot * dt;
        q = q / norm(q);
        w = w + w_dot * dt;
        h = h + h_dot * dt;
        
        X(t, :, :) = quat.A(q) * x_0;
        momenta(t, :) = h;

end