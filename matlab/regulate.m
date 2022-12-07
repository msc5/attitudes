function reg = regulate()
    reg.regulate = @reg;
    reg.plot_errors = @plot_errors;
    reg.plot_momenta = @plot_momenta;
    reg.plot_wheel_momenta = @plot_wheel_momenta;
    reg.decompose = @decompose;
    reg.plot_rotations = @plot_rotations;
    reg.momenta_slope = @momenta_slope;
    reg.analysis = @analysis;
end

function results = reg(params)

    u = util;

    J = params.J;
    w_0 = params.w_0;
    q_c = params.q_c;
    q_time = u.default(params, 'q_time', []);
    M = params.M;
    M_ax = u.default(params, 'M_ax', [1, 0, 0]');
    M_time = u.default(params, 'M_time', []);
    T = params.T;
    k_p = params.k_p;
    k_d = params.k_d;
    n = u.default(params, 'n', 20000);


    vect = vector;
    quat = quaternion;

    % Simulation Parameters
    dt = T / n;
    times = linspace(0, T, n);
    
    if isempty(M_time)
        M_n = [1, n];
    else
        M_n = [find(times >= M_time(1), 1), find(times >= M_time(2), 1)];
    end
    
    if isempty(q_time)
        q_c_n = [1, n];
    elseif q_time == 0
        q_c_n = [0, 0];
    else
        q_c_n = [find(times >= q_time(1), 1), find(times >= q_time(2), 1)];
    end
    M_ref = (M_ax / norm(M_ax)) * M;
    
    errors = zeros(n, 4);
    momenta = zeros(n, 3);
    h_dots = zeros(n, 3);
    errors_mag = zeros(n, 1);

    % Initial Conditions
    h_0 = [0, 0, 0]';
    q_0 = [0, 0, 0, 1]';
    
    % Initial Basis for display
    X = zeros(n, 3, 3);
    x_0 = eye(3);
    X_c = quat.A(q_c) * x_0;

    % Reaction Wheel Simulation
    h = h_0;
    q = q_0;
    w = w_0;
    
    rcs_momentum = 0;
    
    fprintf('Simulating ---------------------------------------------------\n');
    fprintf('Total Time : %d, Moment : %0.5d, k_p : %d, k_d : %d\n', ...
        T, norm(M), k_p, k_d);
    fprintf('%-40s : %.5d Nms\n', 'Total Initial Satellite Momenta', ...
            norm(J * w_0));
        
    for t = 1:n
        
        % Controller
        dq = quat.dq(q, q_c);
        if and(t >= q_c_n(1), t <= q_c_n(2))
            L = - k_p * sign(dq(4)) * dq(1:3) - k_d * w;
        else
            L = [0, 0, 0]';
        end

        if and(t >= M_n(1), t <= M_n(2))
            M = M_ref;
        else
            M = [0, 0, 0]';
        end
        
        % Integrations
        w_dot = - inv(J) * (vect.cross(w) * J * w - L - M);
        h_dot = - vect.cross(w) * h - L;
        q_dot = (1/2) * quat.xi(q) * w;

        q = q + q_dot * dt;
        q = q / norm(q);
        w = w + w_dot * dt;
        h = h + h_dot * dt;
                
        actual = quat.A(q) * x_0;
        
        X(t, :, :) = actual;
        errors(t, :) = dq;
        errors_mag(t) = rad2deg(acos(dot(X_c(:, 1), actual(:, 1)) /...
            (norm(X_c(:, 1)) * norm(X_c(:, 1)))));
        momenta(t, :) = h;
        h_dots(t, :) = h_dot;
        rcs_momentum = rcs_momentum + abs(norm(h_dot * dt));

    end
    
    fprintf('Simulation Complete ------------------------------------------\n\n');
    
    results = {};
    results.params = params;
    results.times = times;
    results.errors = errors;
    results.momenta = momenta;
    results.h_dots = h_dots;
    results.errors_mag = errors_mag;
    results.X = X;
    results.X_c = X_c;
    results.rcs_mom = rcs_momentum;
    
end

function none = analysis(results, wheels)

    times = results.times;
    momenta = results.momenta;
    h_dots = results.h_dots;
    rcs_mom = results.rcs_mom;
    
    pyramid = wheels.pyramid;
    nasa = wheels.nasa;

    reg = regulate;
    pyramid_slopes = reg.momenta_slope(times, pyramid);
    nasa_slopes = reg.momenta_slope(times, nasa);
    fprintf('%-40s : %.5d Nms\n', ...
        'Final Control Momenta Magnitude', norm(momenta(end, :)));
    fprintf('%-40s : %.5d Nms\n', ...
        'Maximum Control Momenta Magnitude', max(vecnorm(momenta')));
    fprintf('%-40s : %.5d Nm\n', ...
        'Final Control Torque Magnitude', norm(h_dots(end, :)));
    fprintf('%-40s : %.5d Nm\n', ...
        'Maximum Control Torque Magnitude', max(vecnorm(h_dots')));
    fprintf('--------------------------------------------------------------\n');
    fprintf('Pyramid Wheel Configuration:\n');
    fprintf('%-40s : %.5d Nms\n', ...
        'Maximum Momenta Magnitude', max(max(abs(pyramid))));
    fprintf('%-40s : %.5d Nms\n', ...
        'Final Momenta Magnitude', max(pyramid(:, end)));
    fprintf('%-40s : %.5d Nms\n', ...
        'Maximum Momenta Accumulation Rate', max(abs(pyramid_slopes)));
    fprintf('--------------------------------------------------------------\n');
    fprintf('NASA Wheel Configuration:\n');
    fprintf('%-40s : %.5d Nms\n', ...
        'Maximum Momenta Magnitude', max(max(abs(nasa))));
    fprintf('%-40s : %.5d Nms\n', ...
        'Final Momenta Magnitude', max(nasa(:, end)));
    fprintf('%-40s : %.5d Nms\n', ...
        'Maximum Momenta Accumulation Rate', max(abs(nasa_slopes)));
    fprintf('--------------------------------------------------------------\n');
    fprintf('%-40s : %.5d Nms\n', ...
        'Total Momentum Cost', rcs_mom);
    fprintf('--------------------------------------------------------------\n');
    none = [];
end

function slopes = momenta_slope(times, momenta)
    n = size(momenta);
    n = n(1);
    slopes = zeros(n, 2);
    for i = 1:n
        slopes(i, :) = polyfit(times(:, 500:end), momenta(i, 500:end), 1);
    end
    slopes = slopes(:, 1);
end

function f = plot_errors(results)
    times = results.times;
    errors = results.errors_mag;
    limit = util().default(results.params, 'error_time', [0, results.params.T]);
    f = figure('visible', 'off');
    f.Position = [400 200 800 300];
    title('Maneuver Errors')
    plot(times, errors);
    xlim(limit);
    grid on;
    ylabel('Error Angle (deg)');
    xlabel('Time (s)');
end

function f = plot_momenta(results)
    times = results.times;
    momenta = results.momenta;
    h_dots = results.h_dots;
    f = figure('visible', 'off');
    f.Position = [400 200 800 600];
    subplot(2, 1, 1);
    hold on;
    plot(times, momenta);
    grid on;
    yline(0, 'k--');
    ylabel('Total Control Momenta (Nms)');
    xlabel('Time (s)');
    legend('h_1', 'h_2', 'h_3', 'Location', 'east');
    subplot(2, 1, 2);
    hold on;
    plot(times, h_dots);
    grid on;
    ylabel('Total Control Torques (Nm)');
    xlabel('Time (s)');
    legend('T_1', 'T_2', 'T_3', 'Location', 'east');
end

function wheels = decompose(results)
    momenta = results.momenta;
    W_pyramid = [1, -1, 0, 0;
                 1, 1, 1, 1;
                 0, 0, 1, -1] * (1 / sqrt(2));
    W_NASA = [1, 0, 0, 1 / sqrt(3); 
              0, 1, 0, 1 / sqrt(3);
              0, 0, 1, 1 / sqrt(3)];
    pyramid = pinv(W_pyramid) * momenta';
    nasa = pinv(W_NASA) * momenta';
    wheels.times = results.times;
    wheels.pyramid = pyramid;
    wheels.nasa = nasa;
end

function f = plot_wheel_momenta(wheels)
    times = wheels.times;
    pyramid = wheels.pyramid;
    nasa = wheels.nasa;
    f = figure('visible', 'off');
    f.Position = [400 200 800 600];
    title('Wheel Momenta Comparison')
    subplot(2, 1, 1);
    plot(times, pyramid);
    grid on;
    yline(0,'k--');
    ylabel({'Pyramid Configuration', 'Wheel Momenta (Nms)'});
    xlabel('Time (s)');
    legend('wheel 1', 'wheel 2', 'wheel 3', 'wheel 4', 'Location', 'east');
    subplot(2, 1, 2);
    plot(times, nasa);
    grid on;
    yline(0,'k--');
    ylabel({'Nasa Configuration', 'Wheel Momenta (Nms)'});
    xlabel('Time (s)');
    legend('wheel 1', 'wheel 2', 'wheel 3', 'wheel 4', 'Location', 'east');
end

function f = plot_rotations(results)
    X = results.X;
    X_c = results.X_c;
    origin = [0, 0, 0]';
    X_0 = squeeze(X(1, :, :));
    X_f = squeeze(X(end, :, :));
    f = figure('visible', 'off');
    f.Position = [400 200 700 600];
    title('Satellite Rotation Trajectory')
    hold on; view(3);
    view(135, 30);
    xlim([-1, 1]); ylim([-1, 1]), zlim([-1, 1]);
    plot3(X(:, 1, 1), X(:, 1, 2), X(:, 1, 3), ...
          X(:, 2, 1), X(:, 2, 2), X(:, 2, 3), ...
          X(:, 3, 1), X(:, 3, 2), X(:, 3, 3));
    quiver3(origin, origin, origin, X_0(:, 1), X_0(:, 2), X_0(:, 3), 'b');
    quiver3(origin, origin, origin, X_c(:, 1), X_c(:, 2), X_c(:, 3), 'm');
    quiver3(origin, origin, origin, X_f(:, 1), X_f(:, 2), X_f(:, 3), 'r');
    legend('X axis rotation', 'Y axis rotation', 'Z axis rotation', ...
        'q_0', 'q_c', 'q_f');
end