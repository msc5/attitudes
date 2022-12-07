function vector = vector()
    vector.cross = @cross;
    vector.sat = @sat;
end

function cross = cross(v)
    cross = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function s = sat(s, ei)
    for i = 1:length(s)
        if s(i) > ei
            s(i) = 1;
        elseif abs(s(i)) <= ei
            s(i) = s(i) / ei;
        elseif s(i) < -ei
            s(i) = -1;
        end
    end
end