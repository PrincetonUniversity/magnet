function [Freq, Error] = IdentificationFrequencyZeroPaddingAndWindowing(Signal, Freq_fft, Ts, resolution, display)
%IdentificationFrequencyZeroPaddingAndWindowing Hann window and zero
%padding for the frequency calculation
% more accurate reulsts
%   Signal - either the voltage or the current matrix
%   Freq_fft - reference frequency obtained with regular FFT
%   Ts - sampling time
%   resolution - desired frequency resolution
%   display - additional plots and messages
%   Freq - interpolated frequency vector
%   Error - error with respect to FFT method
%
%   Zero-padding to increase the resolution of the FFT: https://www.bitweenie.com/listings/fft-zero-padding/
%   Hann window of the sample to avoid spectral leakage https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf

Ndata = length(Signal(:,1));
Nsamples = length(Signal(1,:));
Freq = zeros(Ndata,1); % Initialization

Repeat_times = ceil(1/(Ts*Nsamples)/resolution); % How many zeros to add after the signal
Nsamples_long = Nsamples*Repeat_times;

Freq_max = 1e6; % Maximum frequency to output and plot
fn_max = min(ceil(Freq_max*(Ts*Nsamples)),Nsamples/2);
fn_long_max = fn_max*Repeat_times;

% Hann window
hann_window = hann(Nsamples); % Defining the hann window

for n = 1:Ndata
    sample = Signal(n,:); % Not to carry the (n,:) all around
    % Hann window
    sample_windowed = hann_window'.*sample;
    % Zero-padding
    sample_long = [sample_windowed zeros(1,Nsamples_long-Nsamples)]; % Mean removed as in extreme cases, the mean value may be above the first harmonic becuase of the Hann windown fucntion
    % FFT
    fft_long = GetFFTshort(sample_long, fn_long_max); % Only for even number of samples
    [~, fn_long] = max(fft_long);
    Freq(n) = fn_long/Ts/Nsamples_long;
    
    if display==1 && rem(n,50)==0
        disp(['Frequency algorithm: ', num2str(round(n/Ndata*100,1)), ' % done'])
    end    

    if display==1 && n==round(Ndata/2)
        fft_sample = GetFFTshort(sample, fn_max); % Only for even number of samples
        figure;
        subplot(1,2,1);
        plot((1:2*Nsamples)*Ts*1e6, [sample_windowed zeros(1,Nsamples)], 'k'); % Only an extra sample for the plot
        xlabel('Time [us]');
        ylabel('Amplitude');
        subplot(1,2,2); hold on;
        plot((1:fn_max)/Ts/Nsamples*1e-3, fft_sample, '.b');
        plot((1:fn_long_max)/Ts/Nsamples_long*1e-3, fft_long*Repeat_times, '.k');
        plot(Freq(n)*1e-3, fft_long(fn_long)*Repeat_times, 'ok');
        plot(Freq_fft(n)*1e-3, max(fft_sample), 'or');
        xlabel('Frequency [kHz]');
        ylabel('Amplitude');
        legend('FFT', 'Hann+Zero-Pad');
        set(gca, 'YScale', 'log');
        sgtitle(['Datapoint=', num2str(n)]);
        drawnow();
    end
end

Error = (Freq_fft-Freq)./Freq_fft; % Error with this method

disp(['The average error in frequency is ',num2str(round(mean(abs(Error*100)), 3)),' % and the peak error is ',num2str(round(max(abs(Error*100)), 2)),' %'])

if display==1
    figure;
    subplot(1,3,1)
    plot(Error*100, '.k');
    xlabel('Datapoint');
    ylabel('Frequency mismatch [\%]');
    [~,n_worst] = max(abs(Error));
    sample = Signal(n_worst,:); % Not to carry the (n,:) all around
    fft_sample = GetFFTshort(sample, fn_max); % Only for even number of samples
    sample_windowed = hann_window'.*sample;
    sample_long = [sample_windowed zeros(1,Nsamples_long-Nsamples)]; % Mean removed as in extreme cases, the mean value may be above the first harmonic becuase of the Hann windown fucntion
    fft_long = GetFFTshort(sample_long, fn_long_max); % Module of the fft
    [~, fn] = max(fft_long);
    
    subplot(1,3,2);
    plot((1:2*Nsamples)*Ts*1e6, [sample_windowed zeros(1,Nsamples)], 'k');
    xlabel('Time [us]');
    ylabel('Amplitude');
    subplot(1,3,3); hold on;
    plot((1:fn_max)/Ts/Nsamples*1e-3, fft_sample, '.b');
    plot((1:fn_long_max)/Ts/Nsamples_long*1e-3, fft_long*Repeat_times, '.k');
    plot(Freq(n_worst)*1e-3, fft_long(fn)*Repeat_times, 'ok');
    plot(Freq_fft(n_worst)*1e-3, max(fft_sample), 'or');
    xlabel('Frequency [kHz]');
    ylabel('Amplitude');
    legend('FFT', 'Hann+Zero-Pad');
    set(gca, 'YScale', 'log');
    sgtitle(['Worst-Case for Frequency, Datapoint=', num2str(n_worst)]);
    drawnow();
end

end
