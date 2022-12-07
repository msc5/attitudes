function linear = linear()
    linear.ss = @ss;
end

function [A, B, C] = ss(J, w)
    j = diag(J);
    j_1 = (j(3) - j(2)) / j(1);
    j_2 = (j(1) - j(3)) / j(2);
    j_3 = (j(2) - j(1)) / j(3);
    A_w = [0, j_1 * w(3), j_1 * w(2);
           j_2 * w(3), 0, j_2 * w(1);
           j_3 * w(2), j_3 * w(1), 0];
    A = [zeros(3, 3), eye(3); A_w, zeros(3, 3)];
    B = [zeros(3, 3); eye(3)];
    C = [eye(3), zeros(3, 3)];
end