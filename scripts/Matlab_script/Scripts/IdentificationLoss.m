function Loss = IdentificationLoss(Volt, Curr, Freq, Ts, display)
%IdentificationLoss Find the power losses
%   Volt - voltage matrix (single precision)
%   Curr - voltage matrix (single precision)
%   Freq - switching frequency
%   Ts - sampling time (single precision)
%   display - additional plots and messages
%   Loss - power loss, without averages and for a integer number of cycles
%
%   Losses with this algorithm depend (sometimes heavily) on the frequency
%   identified

Ndata = length(Volt(:,1));
Nsamples = length(Volt(1,:));
Loss = zeros(Ndata,1);

f_sampling = 1/Ts; % Sampling frequency
t_sample = Ts*Nsamples; % Total time of the sample sample (for 10000 samples with a 10e-9 s sampling time, this is 100e-6 s)
for n = 1:Ndata

    volt_sample = Volt(n,:);
    curr_sample = Curr(n,:);
    loss_product = volt_sample.*curr_sample;

    volt_no_avg = volt_sample-mean(volt_sample);
    curr_no_avg = curr_sample-mean(curr_sample);
    loss_no_avgs = volt_no_avg.*curr_no_avg;

    samples_per_cycle = round(f_sampling/Freq(n)); % Number of samples in a switching cycle, rounded
    Ncycles_round = floor(t_sample*Freq(n)); % Number of switching cycles in the sample, floored
    % Removing the last switching cycle if not complete
    volt_no_avg_integer = volt_sample(1:Ncycles_round*samples_per_cycle)-mean(volt_sample(1:Ncycles_round*samples_per_cycle));
    curr_no_avg_integer = curr_sample(1:Ncycles_round*samples_per_cycle)-mean(curr_sample(1:Ncycles_round*samples_per_cycle));
    loss_integer = volt_no_avg_integer.*curr_no_avg_integer;

    if n==round(Ndata/2) && display==1
        figure
        subplot(3,1,1); hold on;
        plot((1:Nsamples)*Ts*1e6, volt_sample, 'r');
        plot((1:Nsamples)*Ts*1e6, volt_no_avg, 'b');
        plot((1:Ncycles_round*samples_per_cycle)*Ts*1e6, volt_no_avg_integer, 'k');
        xlabel('Time [us]');
        ylabel('Voltage [V]');
        subplot(3,1,2); hold on;
        plot((1:Nsamples)*Ts*1e6, curr_sample, 'r');
        plot((1:Nsamples)*Ts*1e6, curr_no_avg, 'b');
        plot((1:Ncycles_round*samples_per_cycle)*Ts*1e6, curr_no_avg_integer, 'k');
        xlabel('Time [us]');
        ylabel('Current [A]');
        subplot(3,1,3); hold on;
        plot((1:Nsamples)*Ts*1e6, loss_product, 'r');
        plot((1:Nsamples)*Ts*1e6, loss_no_avgs, 'b');
        plot((1:Ncycles_round*samples_per_cycle)*Ts*1e6, loss_integer, 'k');
        xlabel('Time [us]');
        ylabel('Power [W]');
        legend('Direct product', 'Product without averages', 'Only for an integer number of cycles')
        sgtitle(['Datapoint=', num2str(n)]);
        drawnow();
    end

    Loss(n) = mean(loss_integer);
end

% Plot the power loss
if display==1
    figure;
    plot(Loss, '.k');
    xlabel('Datapoint');
    ylabel('Loss [W]');
    set(gca, 'YScale', 'log');
    drawnow();
end

end
