function quaternion = quaternion()
    quaternion.q = @q;
    quaternion.psi = @psi;
    quaternion.xi = @xi;
    quaternion.cross = @cross;
    quaternion.dot = @dot;
    quaternion.dq = @dq;
    quaternion.I = @I;
    quaternion.inv = @inv;
    quaternion.conj = @conj;
    quaternion.A = @A;
end

function q = q(e, t)
    q = [e * sin(t / 2); cos(t / 2)];
end

function psi = psi(q)
    vec = vector;
    v = q(1:3);
    psi = [q(4) * eye(3) - vec.cross(v); -v'];
end

function xi = xi(q)
    vec = vector;
    v = q(1:3);
    xi = [q(4) * eye(3) + vec.cross(v); -v'];
end

function cross = cross(q)
    cross = [psi(q), q];
end

function dot = dot(q)
    dot = [xi(q), q];
end

function A = A(q)
    A = xi(q)' * psi(q);
end

function dq = dq(q, q_c)
    dq = cross(q) * q_c;
end

function conj = conj(q)
    conj = [-q(1:3); q(4)];
end

function inv = inv(q)
    inv = conj(q) / (norm(q))^2;
end

function I = I()
    I = [0, 0, 0, 1]';
end

