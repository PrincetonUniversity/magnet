function [output, error] = SingleCycleInterpolation(input, N_in, Ts_in, N_out, Ts_out, display)
%SingleCycleInterpolation Spline interpolation of the current or voltage measurement
%   input - input sample
%   N_in - number of samples of the input vector
%   Ts_in - Sampling time of the input vector
%   N_out - Number of samples of the output vector
%   Ts_out - Switching period time
%   output - output sample
%   error - difference between the first and last cycles
%
%   Fisrt, the input cycle is divided in switching cycles, that are then
%   averaged.

T_in = Ts_in*N_in; % Input total time
T_out = Ts_out*N_out; % Output total time (switching period)

Ncycles = T_in/T_out; % Number of cycles at the tested frequency in the total sample
Nin_per_cycle = N_in/Ncycles;
in_time = (0:N_in-1)*Ts_in; % Time sequence
out_time = (0:N_out-1)*Ts_out;

out_iteration = zeros(floor(Ncycles),N_out); % Initialization
for n_cycle = 1:floor(Ncycles) % The number of the cycle
    sample_start = max(floor(Nin_per_cycle*(n_cycle-1))-1,1);
    sample_end = min(sample_start+ceil(Nin_per_cycle)+1,N_in);
    out_time_iteration = out_time+(n_cycle-1)*T_out; % The vector is displaced the right ammount every time
    out_iteration(n_cycle,:) = spline(in_time(sample_start:sample_end), input(sample_start:sample_end), out_time_iteration);
    % There is no need to define the sample_start and sample_end for the
    % spline function but it may speed up the computation time
end
output = mean(out_iteration);
% TODO: Define the error as mean square rather than a vector.
error = mean(mean(abs(repmat(output,floor(Ncycles),1)-out_iteration(n_cycle,:))));
if display==1 % Plot a specific voltage and current
    figure; hold on;
    plot(in_time*1e6, input, 'r')
    plot(out_time*1e6, output, 'ob');
    for n_cycle = 1:floor(Ncycles) % The number of the cycle 
        out_time_iteration = out_time+(n_cycle-1)*T_out; % The vector is displaced the right ammount every time
        out_iteration(n_cycle,:) = spline(in_time, input, out_time_iteration);
        plot(out_time_iteration*1e6, out_iteration(n_cycle,:), '.b');
        plot([n_cycle*T_out*1e6 n_cycle*T_out*1e6], [min(input) max(input)], '--k')
    end
    xlabel('Time [us]');
    ylabel('Voltage or Current [V or A]');
    legend('Input', 'Output');
    drawnow();
end

end