function wmap = wmap()
    wmap.commands = @commands;
    wmap.pos = @pos;
    wmap.q_0 = @q_0;
    wmap.a = @a;
end

function q_0 = q_0(Phi, q_c)
    quat = quaternion;
    q_0 = quat.cross([0, 0, sin(Phi / 2), cos(Phi / 2)]') * q_c;
end

function [a1, a2, a3] = a(phi, theta, psi)
    a1 = cos(phi) * cos(theta) * cos(psi)...
        - sin(phi) * cos(theta)^2 * sin(psi) + sin(phi) * sin(theta)^2;
    a2 = sin(phi) * cos(theta) * cos(psi)...
        + cos(phi) * cos(theta)^2 * sin(psi) - cos(phi) * sin(theta)^2;
    a3 = cos(theta) * sin(theta) * (sin(psi) + 1);
end

function [x, y] = pos(phi, theta, psi)
    [a1, a2, a3] = a(phi, theta, psi);
    x = a1 / (1 + a3);
    y = a2 / (1 + a3);
end

function [q_c, w_c, w_c_dot] = commands(phi, theta, psi)    
    phi_dot = 0.001745;
    psi_dot = 0.04859;
    q_c = [...
        sin(theta / 2) * cos((phi - psi) / 2);
        sin(theta / 2) * sin((phi - psi) / 2);
        cos(theta / 2) * sin((phi + psi) / 2);
        cos(theta / 2) * cos((phi + psi) / 2)];
    w_c = [...
        phi_dot * sin(theta) * sin(psi);
        phi_dot * sin(theta) * cos(psi);
        psi_dot];
    w_c_dot = [...
        sin(theta) * cos(psi);
        - sin(theta) * sin(psi);
        0] * phi_dot * psi_dot;
end