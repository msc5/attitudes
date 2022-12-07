function util = util()
    util.default = @default;
    util.save_plots = @save_plots;
    util.show_plots = @show_plots;
    util.reset = @reset;
end

function val = default(s, key, default)
    if isfield(s, key)
        val = s.(key);
    else
        val = default;
    end
end

function [u, reg, quat] = reset()
    u = util;
    reg = regulate;
    quat = quaternion;
end

function none = show_plots(show_bool, plots)
    if show_bool
        for i = 1:length(plots)
            set(plots(i), 'visible', 'on');
        end
    end
end

function none = save_plots(save_bool, name, plots)
    names = {'Errors', 'Momenta', 'Wheel_Momenta', 'Rotations'};
    if save_bool
        for i = 1:length(plots)
            saveas(plots(i), ['figures/' name '_' names{i} '.png']);
        end
    end
end