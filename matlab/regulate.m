function reg = regulate()
    reg.regulate = @reg;
    reg.plot_momenta = @plot_momenta;
    reg.plot_wheel_momenta = @plot_wheel_momenta;
    reg.decompose = @decompose;
    reg.plot_rotations = @plot_rotations;
end

function [times, errors, momenta, X] = reg(J, w_0, q_c, M, T)
    vect = vector;
    quat = quaternion;

    % Simulation Parameters
    n = 10000;
    dt = T / n;
    times = linspace(0, T, n);

    errors = zeros(n, 4);
    momenta = zeros(n, 3);

    % Initial Conditions
    h_0 = [0, 0, 0]';
    q_0 = [0, 0, 0, 1]';

    % Controller Parameters
    k_p = 10;
    k_d = 150;
    
    % Initial Basis for display
    X = zeros(n, 3, 3);
    x_0 = eye(3);

    % Reaction Wheel Simulation
    h = h_0;
    q = q_0;
    w = w_0;
    for t = 1:n
        
        % Controller
        dq = quat.dq(q, q_c);
        L = - k_p * sign(dq(4)) * dq(1:3) - k_d * w;

        % Integration
        w_dot = - inv(J) * (vect.cross(w) * J * w - L - M);
        h_dot = - vect.cross(w) * h - L;
        q_dot = (1/2) * quat.xi(q) * w;

        q = q + q_dot * dt;
        q = q / norm(q);
        w = w + w_dot * dt;
        h = h + h_dot * dt;
        
        X(t, :, :) = quat.A(q) * x_0;

        errors(t, :) = dq;
        momenta(t, :) = h;

    end
end

function f = plot_momenta(times, errors, momenta)
    f = figure;
    f.Position = [400 200 1200 800];
    title('Maneuver Errors and Total Wheel Momenta')
    subplot(2, 1, 1);
    grid on;
    plot(times, errors);
    yline(0,'k--');
    yline(1,'k--');
    ylabel('Quaternion Errors');
    xlabel('Time (s)');
    legend('q_1', 'q_2', 'q_3', 'q_4', 'location', 'best');
    subplot(2, 1, 2);
    grid on;
    plot(times, momenta);
    yline(0,'k--');
    ylabel('Total Wheel Momenta (Nms)');
    xlabel('Time (s)');
    legend('h_1', 'h_2', 'h_3', 'location', 'best');
end

function [pyramid, nasa] = decompose(momenta)
    W_pyramid = [1, -1, 0, 0;
                 1, 1, 1, 1;
                 0, 0, 1, -1] * (1 / sqrt(2));
    W_NASA = [1, 0, 0, 1 / sqrt(3); 
              0, 1, 0, 1 / sqrt(3);
              0, 0, 1, 1 / sqrt(3)];
    pyramid = pinv(W_pyramid) * momenta';
    nasa = pinv(W_NASA) * momenta';
end

function f = plot_wheel_momenta(times, pyramid, nasa)
    f = figure;
    f.Position = [400 200 1200 800];
    title('Wheel Momenta Comparison')
    grid on;
    subplot(2, 1, 1);
    plot(times, pyramid);
    yline(0,'k--');
    ylabel({'Pyramid Configuration', 'Wheel Momenta (Nms)'});
    xlabel('Time (s)');
    legend('w_1', 'w_2', 'w_3', 'w_4', 'location', 'best');
    subplot(2, 1, 2);
    plot(times, nasa);
    yline(0,'k--');
    ylabel({'Nasa Configuration', 'Wheel Momenta (Nms)'});
    xlabel('Time (s)');
    legend('w_1', 'w_2', 'w_3', 'w_4', 'location', 'best');

    fprintf('Pyramid:   %.3f\n', norm(pyramid(:, end)));
    fprintf('NASA:      %.3f\n', norm(nasa(:, end)));
end

function f = plot_rotations(X)
    origin = [0, 0, 0]';
    X_0 = squeeze(X(1, :, :));
    X_f = squeeze(X(end, :, :));

    f = figure;
    f.Position = [400 200 900 800];
    title('Satellite Rotation Trajectory')
    hold on; view(3);
    xlim([-1, 1]); ylim([-1, 1]), zlim([-1, 1]);
    plot3(X(:, 1, 1), X(:, 1, 2), X(:, 1, 3), ...
          X(:, 2, 1), X(:, 2, 2), X(:, 2, 3), ...
          X(:, 3, 1), X(:, 3, 2), X(:, 3, 3));
    quiver3(origin, origin, origin, X_0(:, 1), X_0(:, 2), X_0(:, 3), 'b');
    quiver3(origin, origin, origin, X_f(:, 1), X_f(:, 2), X_f(:, 3), 'r');
    legend('X axis rotation', 'Y axis rotation', 'Z axis rotation', ...
        'q_0', 'q_f');
end