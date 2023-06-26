function [dP, dN] = GetDutyCycles(sequences, f, Ts, Nthresholds, remove_pk_frac, display)
%GetDutyCycles Find the positive and negative duty cycle
%   sequences - voltage matrix (single precision)
%   f - frequency of the signal
%   Ts - sampling time (single precision)
%   Nthresholds - number of threshold voltages evaluated
%   remove_pk_frac - fraction of minimum and maximum values to remove the
%   switching noise
%   display - additional plots and messages
%   dP - fraction of the period with the maximum possitive voltage
%   dN - fraction of the period with the maximum negative voltage
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

Ndata = length(sequences(:,1));
Nsamples = length(sequences(1,:));

dP = zeros(Ndata,1); dN = zeros(Ndata,1); % Initialization


for n = 1:Ndata

    ts = Ts(n);
    sample = sequences(n,:); % Not to carry the (n,:) all around

    % Remove the switching noise
    sample_no_peaks = sample;
    [~, topidx] = sort(sample_no_peaks, 'descend');
    remidx=topidx(1:round(Nsamples*remove_pk_frac));
    sample_no_peaks(remidx)=[];
    [~, botidx] = sort(sample_no_peaks, 'ascend');
    remidx=botidx(1:round(Nsamples*remove_pk_frac));
    sample_no_peaks(remidx)=[];

    n_of_th = 0; % Initialization
    fraction_above_th = zeros(Nthresholds+1,1); % Initialization
    for amplitude_th = min(sample_no_peaks):(max(sample_no_peaks)-min(sample_no_peaks))/Nthresholds:max(sample_no_peaks) % Do a sweep in the treshold, "Nthresholds" thresholds between Vmin and Vmax
        n_of_th = n_of_th+1;
        fraction_above_th(n_of_th) = sum(sample_no_peaks>amplitude_th)/length(sample_no_peaks);
    end
    %% TO DO FIX THE EXTREMES

    duty_top = fraction_above_th(round(length(fraction_above_th)/4));
    % One of the duty cycles of interest is situated in the first quarter of the threshold 
    duty_bot = fraction_above_th(round(3*length(fraction_above_th)/4));
    % The other in the third quarter, to better understand what is happening plot Nthresholds.

    dP(n)=duty_bot; % The positive duty cycle is the percentage above the maximum threshold for which there are values
    dN(n)=1-duty_top;

    if n==round(Ndata/2) && display==1
        figure;
        subplot(1,2,1);
        hold on;
        plot((1:length(sample))*ts*1e6, sample, 'r');
        plot((1:length(sample_no_peaks))*ts*1e6, sample_no_peaks, 'k');
        for amplitude_th = min(sample_no_peaks):(max(sample_no_peaks)-min(sample_no_peaks))/Nthresholds:max(sample_no_peaks) % Do a sweep in the treshold, "Nthresholds" thresholds between Vmin and Vmax
            plot([1 length(sample_no_peaks)]*ts*1e6, [amplitude_th amplitude_th], '--k');
        end
        xlabel('Time [us]');
        ylabel('Voltage [V]');
        subplot(1,2,2)
        hold on;
        plot(fraction_above_th, 'k');
        plot([round(length(fraction_above_th)/4), round(3*length(fraction_above_th)/4)], [duty_top, duty_bot],'ok')
        plot([round(length(fraction_above_th)/4), round(length(fraction_above_th)/4)], [duty_top, 1],'k')
        plot([round(3*length(fraction_above_th)/4), round(3*length(fraction_above_th)/4)], [0, duty_bot],'k')
        text(round(length(fraction_above_th)/4), 1-(1-duty_top)/2, '$\leftarrow d_1$', 'interpreter', 'latex');
        text(round(3*length(fraction_above_th)/4), duty_bot/2, '$\leftarrow d_3$', 'interpreter', 'latex');
        xlabel('Threshold sweep point');
        ylabel('Fraction above threshold');
        sgtitle(['Datapoint=', num2str(n)]);
        drawnow();
    end
end

% Plot the duty cycles
if display==1
    figure; hold on
    plot(dP, '.r');
    plot(dN, '.b');
    xlabel('Datapoint');
    ylabel('Duty cycle');
    legend('$d_p$', '$d_n$')
    drawnow();
end

end