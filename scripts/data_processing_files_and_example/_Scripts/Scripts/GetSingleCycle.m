function [out_cycle, nrmse] = GetSingleCycle(sequences, f, Ts, Ncycle, display, display_aux)
%GetSingleCycle saves a single cycle for the voltage and current at fixed
%time step
%   sequences - input sequence matrix
%   f - frequency vector (critical parameter)
%   Ts - sampling time vector
%   Ncycle - number of samples for the output
%   display - plots and messages
%   display_aux - additional plots and messages
%   out_cycle - single cycle waveforms
%   nrmse - normalized RMSE between each interpolated output and the single cycle
%   output
%
%   Generates the single cycle signal for a given frequency (therefore,
%   with a different time step per datapoint). To do so, first, the
%   waveform is divided into parts based on the input frequency. Incomplete
%   cycles at the end of the sample are discarded. Then the different parts
%   are averaged, and finally the sample is interpolated to have the
%   desired number of points.

Ndata = length(sequences(:,1)); % Number of datapoints
Nsamples = length(sequences(1,:)); % Number of samples per datapoint
out_cycle = zeros(Ndata,Ncycle); nrmse = zeros(Ndata,1); % Initialization

for n = 1:Ndata

    ts = Ts(n);
    Tcycle = 1/f(n); % Period
    ts_cycle = Tcycle/Ncycle; % Sampling time of the switching cycle

    sample = sequences(n,:);

    [out_cycle(n,:), nrmse(n)] = SingleCycleInterpolation(sample, Nsamples, ts, Ncycle, ts_cycle, 0);

    if n==round(Ndata/2) && display==1 % Plot a specific voltage and current

    figure
    subplot(1,2,1); hold on
        plot((0:floor(Tcycle/ts)-1)*ts*1e6, sample(1:floor(Tcycle/ts)), '.r')
        plot((0:Ncycle-1)*ts_cycle*1e6, out_cycle(n,:), '.k');
        xlabel('Time [us]');
        ylabel('Amplitude');
        legend('First cycle', 'Single cycle');
     subplot(1,2,2); hold on
        plot((0:Nsamples-1)*ts*1e6, sample, '.r')
        plot((0:Ncycle-1)*ts_cycle*1e6, out_cycle(n,:), '.k');
        xlabel('Time [us]');
        ylabel('Amplitude');
        legend('Measured', 'Single cycle');
    sgtitle(['Datapoint=', num2str(n), '; NRMSE=', num2str(round(nrmse(n)*100,2)), '\%']);
    end
  
    if display==1 && rem(n,200)==0  % Every 100 calculations
        disp(['Single cycle algorithm: ', num2str(round(n/Ndata*100,1)), '% done'])
    end
end

% Plot the RMSE for both current and voltage
if display_aux==1
    figure;
    hold on;
    plot(nrmse*100, '.k');
    xlabel('Datapoint');
    ylabel('NRMSE');
    drawnow();
    disp(['The average NRMSE is ', num2str(round(mean(nrmse)*100, 3)), '% and the peak NRMSE is ', num2str(round(max(nrmse)*100, 2)), '%'])
end

end
