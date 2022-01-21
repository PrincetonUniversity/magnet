function [DutyP, DutyN] = IdentificationDutyCycles(signal, Freq, Ts, resolution, Nthresholds, display)
%IdentificationDutyCycles Find the positive and negative duty cycle
%   signal - voltage matrix (single precision)
%   Freq - frequency of the signal
%   Ts - sampling time (single precision)
%   resolution - resolution for the duty cycle (0.1 if not specified)
%   Nthresholds - number of threshold voltages evaludated (30 if not specified)
%   display - additional plots and messages
%   DutyP - fraction of the period with the maximum possitive voltage
%   DutyN - fraction of the period with the maximum negative voltage
%
% Only valid for Trapezoidal waveforms with d2=d4.
% The idea is to do a sweep of "threshold" voltages between the maximum and
% minimum, and for each threshold, count how many voltage samples are above
% the threshold, then, dividing this number over the total number of
% samples, an idea of the shape of the waveform is obtained.
% Then we delete those values equal to 0 or 1, as they do not provide any information.
% Finally, it is just a matter of finding the the value above Vc and below Vc,
% due to symetry (only if noise is more or less symetric too) we can take
% the value at 1/4 of the treshold length and at 3/4th of the threshold
% length, however this is only valid for d2=d4

Ndata = length(signal(:,1));
Nsamples = length(signal(1,:));

DutyP = zeros(Ndata,1); DutyN = zeros(Ndata,1); % Initialization

f_sampling = 1/Ts;
t_sample = Ts*Nsamples; % Total time of the sample sample (for 10000 samples with a 10e-9 s sampling time, this is 100e-6 s)

for n = 1:Ndata

    sample = signal(n,:); % Not to carry the (n,:) all around
    sample_max = max(sample);
    sample_min = min(sample);
    sample_pkpk = sample_max-sample_min;

    f_cycle=Freq(n);
    samples_per_cycle = round(f_sampling/f_cycle); % Number of samples in a cycle, rounded
    Ncycles_round = round(t_sample*Freq(n)); % Number of switching cycles in the sample, rounded

    sample_without_edges = sample(samples_per_cycle:(Ncycles_round-1)*samples_per_cycle-1); % Removing the first and last cycles

    n_of_th = 0; % Initialization
    fraction_above_th_real = zeros(Nthresholds+1,1); % Initialization
    for amplitude_th = sample_min:sample_pkpk/Nthresholds:sample_max % Do a sweep in the treshold, "Nthresholds" thresholds between Vmin and Vmax
        n_of_th = n_of_th+1;
        %fraction_above_th_real(n_of_th) = length(find(sample_without_edges>amplitude_th))/((Ncycles_round-2)*samples_per_cycle); % Find which fraction of the voltage data is above that threshold
        % Find which fraction of the voltage data is above that threshold
        fraction_above_th_real(n_of_th) = sum(sample_without_edges>amplitude_th)/((Ncycles_round-2)*samples_per_cycle);
    end
    fraction_above_th = round(fraction_above_th_real/resolution)*resolution; % Then round it to the selected resolution
    fraction_above_th(fraction_above_th>1-resolution) = []; fraction_above_th(fraction_above_th<resolution) = [];
    
    duty_top = fraction_above_th(round(length(fraction_above_th)/4));
    % One of the duty cycles of interest is situated in the first quarter of the threshold 
    duty_bot = fraction_above_th(round(3*length(fraction_above_th)/4));
    % The other in the third quarter, to better understand what is happening plot Nthresholds.

    DutyP(n)=duty_bot; % The positive duty cycle is the percentage above the maximum threshold for which there are values
    DutyN(n)=1-duty_top;

    if n==round(Ndata/2) && display==1
        figure; hold on;
        plot(fraction_above_th, 'k');
        plot([round(length(fraction_above_th)/4), round(3*length(fraction_above_th)/4)], [duty_top, duty_bot],'ok')
        plot([round(length(fraction_above_th)/4), round(length(fraction_above_th)/4)], [duty_top, 1],'k')
        plot([round(3*length(fraction_above_th)/4), round(3*length(fraction_above_th)/4)], [0, duty_bot],'k')
        text(round(length(fraction_above_th)/4), 1-(1-duty_top)/2, '$\leftarrow d_1$', 'interpreter', 'latex');
        text(round(3*length(fraction_above_th)/4), duty_bot/2, '$\leftarrow d_3$', 'interpreter', 'latex');
        xlabel('Threshold sweep point');
        ylabel('Fraction above threshold');
        title(['Datapoint=', num2str(n)]);
        drawnow();
    end
end

% Plot the duty cycles
if display==1
    figure; hold on
    plot(DutyP, '.r');
    plot(DutyN, '.b');
    xlabel('Datapoint');
    ylabel('Duty Cycle');
    legend('Positive Duty Cycle', 'Negative Duty Cycle')
    drawnow();
end

end