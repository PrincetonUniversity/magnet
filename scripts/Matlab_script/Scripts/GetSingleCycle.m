function [Volt_Cycle, Curr_Cycle, Volt_Error, Curr_Error] = GetSingleCycle(Volt, Curr, Freq, Ts, Ncycle, display)
%GetSingleCycle saves a single cycle for the voltage and current at fexed
%time step
%   Volt - Voltage matrix
%   Curr - Current matrix
%   Freq - Frequency vector (critical parameter)
%   Ts - Sampling time
%   Ncycle - Number of samples for the output
%   display - additional plots and messages
%   Volt_Cycle - single cycle voltage waveform
%   Curr_Cycle - single cycle current waveform
%   Error_Volt - error between the first and last sample with respect to
%   the average, for the voltage
%   Error_Curr - error between the first and last sample with respect to
%   the average, for the current
%
%   Generates the single cycle signal for a given frequency (therefore,
%   with a different time step per datapoint). To do so, first, the
%   waveform is divided into parts based on the input frequency. Incomplete
%   cycles at the end of the sample are discarded. Then the different parts
%   are averaged, and finally the sample is interpolated to haave the
%   desired number of points. Finally, the average is removed

Ndata = length(Volt(:,1)); % Number of datapoints
Nsamples = length(Volt(1,:)); % Number of samples per datapoint
Volt_Cycle = zeros(Ndata,Ncycle); Curr_Cycle = zeros(Ndata,Ncycle); % Initialization
Volt_Error = zeros(Ndata,1); Curr_Error = zeros(Ndata,1);

for n = 1:Ndata

    sample_volt = Volt(n,:);
    sample_curr = Curr(n,:); % Not to carry the (n,:) all around
    Tcycle = 1/Freq(n); % Period
    
    Ts_cycle = Tcycle/Ncycle; % Sampling time of the switching cycle
    [volt_cycle_shifted, volt_error] = SingleCycleInterpolation(sample_volt, Nsamples, Ts, Ncycle, Ts_cycle, 0);
    [curr_cycle_shifted, curr_error] = SingleCycleInterpolation(sample_curr, Nsamples, Ts, Ncycle, Ts_cycle, 0);

    if n==round(Ndata/2) && display==1 % Plot a specific voltage and current
        
    figure
    subplot(2,2,1); hold on
        plot((0:floor(Tcycle/Ts)-1)*Ts*1e6, sample_volt(1:floor(Tcycle/Ts)), 'r')
        plot((0:Ncycle-1)*Ts_cycle*1e6, volt_cycle_shifted, '.b');
        plot([0 Ncycle-1]*Ts_cycle*1e6, [volt_error volt_error], 'k');
        xlabel('Time [us]');
        ylabel('Voltage [V]');
        legend('First cycle', 'Output', 'Error between cycles');
    subplot(2,2,2) ;hold on
        plot((0:floor(Tcycle/Ts)-1)*Ts*1e6, sample_curr(1:floor(Tcycle/Ts)), 'r')
        plot((0:Ncycle-1)*Ts_cycle*1e6, curr_cycle_shifted, '.b');
        plot([0 Ncycle-1]*Ts_cycle*1e6, [curr_error curr_error], 'k');
        xlabel('Time [us]');
        ylabel('Current [A]');
        legend('First cycle', 'Output', 'Error between cycles');
     subplot(2,2,3); hold on
        plot((0:Nsamples-1)*Ts*1e6, sample_volt, 'r')
        plot((0:Ncycle-1)*Ts_cycle*1e6, volt_cycle_shifted, '.b');
        xlabel('$t$ [us]');
        ylabel('$v$ [V]');
        legend('Measured', 'Single Cycle');
    subplot(2,2,4); hold on
        plot((0:Nsamples-1)*Ts*1e6, sample_curr, 'r')
        plot((0:Ncycle-1)*Ts_cycle*1e6, curr_cycle_shifted, '.b');
        xlabel('$t$ [us]');
        ylabel('$i$ [A]');
        legend('Measured', 'Single Cycle');
    sgtitle(['Datapoint=', num2str(n)]);
    drawnow();
    end
    
    % We can modify the vector so it start with the maximum voltage
    [~, Nstart] = max(volt_cycle_shifted);
    volt_cycle_shifted_double = [volt_cycle_shifted, volt_cycle_shifted];
    curr_cycle_shifted_double = [curr_cycle_shifted, curr_cycle_shifted]; % Repeat the vector

    volt_cycle = zeros(1,Ncycle); curr_cycle = zeros(1,Ncycle); % Initialization
    volt_cycle(1:Ncycle) = volt_cycle_shifted_double(Nstart:Ncycle+Nstart-1); % Displaced so the waveform starts where it is supposed to start
    curr_cycle(1:Ncycle) = curr_cycle_shifted_double(Nstart:Ncycle+Nstart-1);   

    % Saving the data
    Volt_Error(n) = volt_error/mean(abs(volt_cycle));
    Curr_Error(n) = curr_error/mean(abs(curr_cycle));
    
    Volt_Cycle(n,:) = volt_cycle-mean(volt_cycle);
    Curr_Cycle(n,:) = curr_cycle-mean(curr_cycle);
    
    if display==1 && rem(n,50)==0
        disp(['Single cycle algorithm: ', num2str(round(n/Ndata*100,1)), ' % done'])
    end
end

if display==1
    [~,n_worst_volt] = max(abs(Volt_Error));
    sample_volt = Volt(n_worst_volt,:);
    Tcycle = 1/Freq(n_worst_volt); % Period
    Ts_cycle = Tcycle/Ncycle; % Sampling time of the switching cycle
    [volt_cycle_shifted, volt_error] = SingleCycleInterpolation(sample_volt, Nsamples, Ts, Ncycle, Ts_cycle, 1);
    figure;
    subplot(1,2,1)
    plot(Volt_Error*100, '.k');
    xlabel('Datapoint');
    ylabel('Mismatch in voltage [\%]');
    subplot(1,2,2); hold on
    plot((0:floor(Tcycle/Ts)-1)*Ts*1e6, sample_volt(1:floor(Tcycle/Ts)), 'r')
    plot((0:Ncycle-1)*Ts_cycle*1e6, volt_cycle_shifted, '.b');
    plot([0 Ncycle-1]*Ts_cycle*1e6, [volt_error volt_error], 'k');
    xlabel('Time [us]');
    ylabel('Voltage [V]');
    legend('First cycle', 'Output', 'Error between cycles');
    sgtitle(['Worst-Case for Voltage, Datapoint=', num2str(n_worst_volt)]);
    drawnow();
    
    [~,n_worst_curr] = max(abs(Curr_Error));
    sample_curr = Curr(n_worst_curr,:);
    Tcycle = 1/Freq(n_worst_curr); % Period
    Ts_cycle = Tcycle/Ncycle; % Sampling time of the switching cycle
    [curr_cycle_shifted, curr_error] = SingleCycleInterpolation(sample_curr, Nsamples, Ts, Ncycle, Ts_cycle, 0);
    figure;
    subplot(1,2,1)
    plot(Curr_Error*100, '.k');
    xlabel('Datapoint');
    ylabel('Mismatch in current [\%]');
    subplot(1,2,2); hold on
    plot((0:floor(Tcycle/Ts)-1)*Ts*1e6, sample_curr(1:floor(Tcycle/Ts)), 'r')
    plot((0:Ncycle-1)*Ts_cycle*1e6, curr_cycle_shifted, '.b');
    plot([0 Ncycle-1]*Ts_cycle*1e6, [curr_error curr_error], 'k');
    xlabel('Time [us]');
    ylabel('Current [A]');
    legend('First cycle', 'Output', 'Error between cycles');
    sgtitle(['Worst-Case for Current, Datapoint=', num2str(n_worst_curr)]);
    drawnow();
end
disp(['The average error in voltage vs the voltage (avg of the abs) is ', num2str(round(mean(abs(Volt_Error))*100, 3)), ' % and the peak error is ', num2str(round(max(abs(Volt_Error))*100, 2)), ' %'])
disp(['The average error in current vs the current (avg of the abs) is ', num2str(round(mean(abs(Curr_Error))*100, 3)), ' % and the peak error is ', num2str(round(max(abs(Curr_Error))*100, 2)), ' %'])

end