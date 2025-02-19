function THD = GetTHD(sequences, Ts, f_max, display)
% Gets the THD based on the harmonic content
%   Signal - either the voltage or the current matrix (single precision)
%   Ts - sampling time (single precision)
%   f_max - maximum frequency to output and plot
%   display - additional plots and messages
%   THD - THD of the signal
%
%   The frequency resolution is given by 1/(Nsamples*Tsampling)
%   Only valid if the first harmonic is the largest

Ndata = length(sequences(:,1));
Nsamples = length(sequences(1,:));
THD = zeros(Ndata,1); % Initialization

for n = 1:Ndata
    ts = Ts(n);
    fn_max = min(ceil(f_max*(ts*Nsamples)),Nsamples/2);
    sample = sequences(n,:); % Not to carry the (n,:) all around
    fft_sample = FFTshort(sample, fn_max); % Only for even number of samples
    fundamental = max(fft_sample);
    THD(n) = sqrt((sum(fft_sample.^2)-fundamental^2))/fundamental;
end

% Plot the frequency
if display==1
    figure;
    plot(THD*100, '.k');
    xlabel('Datapoint');
    ylabel('THD [\%]');
    drawnow();
end

end
