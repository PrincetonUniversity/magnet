function Freq = IdentificationFrequency(Signal, Ts, display)
%IdentificationFrequency Find the frequency through the fft (Modified from Haoran's code)
%   Signal - either the voltage or the current matrix (single precision)
%   Ts - sampling time (single precision)
%   display - additional plots and messages
%   Freq - fft frequency vector with the highest harmonic per datapoint
%
%   The frequency resolution is given by 1/(Nsamples*Tsampling)
%   Only valid if the first harmonic is the largest

Ndata = length(Signal(:,1));
Nsamples = length(Signal(1,:));
Freq = zeros(Ndata,1); % Initialization
Freq_max = 1e6; % Maximum frequency to output and plot
fn_max = min(ceil(Freq_max*(Ts*Nsamples)),Nsamples/2);
for n=1:Ndata
    sample = Signal(n,:); % Not to carry the (n,:) all around
    fft_sample = GetFFTshort(sample, fn_max); % Only for even number of samples
    [~, fn] = max(fft_sample);
    Freq(n) = fn/Ts/Nsamples; % In Hz
    if n==round(Ndata/2) && display==1
        figure;
        subplot(1,2,1);hold on;
        plot((1:Nsamples)*Ts*1e6,sample, 'k');
        xlabel('Time [us]');
        ylabel('Amplitude');
        subplot(1,2,2);hold on;
        plot((1:fn_max)/Ts/Nsamples*1e-3, fft_sample, '.k');
        plot(fn/Ts/Nsamples*1e-3, fft_sample(fn), 'ok');
        xlabel('Frequency [kHz]');
        ylabel('Amplitude');
        set(gca, 'YScale', 'log');
        sgtitle(['Datapoint=', num2str(n)]);
        drawnow();
    end
end

% Plot the frequency
if display==1
    figure;
    plot(Freq*1e-3, '.k');
    xlabel('Datapoint');
    ylabel('Frequency [kHz]');
    drawnow();
end

end
