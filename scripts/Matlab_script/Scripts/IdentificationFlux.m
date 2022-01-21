function Flux = IdentificationFlux(Signal, Freq, Ts, display)
%IdentificationFlux Find the AC through the fft (Modified from Haoran's code)
%   Sample - voltage matrix over Nturns over Ae
%   Freq - frequency of the voltage sample
%   Ts - sampling time (single precision)
%   display - additional plots and messages
%   Flux - AC amplitude flux vector, in T
%
%   Flux density calcualted based on all the switching cycles but the first and last,
%   moving average flux removed with a switching cycle span.

Ndata = length(Signal(:,1));
Nsamples = length(Signal(1,:));
Flux = zeros(Ndata,1); % Initialization

f_sampling = 1/Ts; % Sampling frequency
t_sample = Ts*Nsamples; % Total time of the sample sample (for 10000 samples with a 10e-9 s sampling time, this is 100e-6 s)

for n = 1:Ndata

    sample = Signal(n,:); % Not to carry the (n,:) all around

    samples_per_cycle = round(f_sampling/Freq(n)); % Number of samples in a switching cycle, rounded
    Ncycles_round = round(t_sample*Freq(n)); % Number of switching cycles in the sample, rounded

    % Flux identification
    flux_raw = cumtrapz(sample)*Ts;
    flux_offset = movmean(flux_raw, samples_per_cycle); % Filtering at the switching frequency
    flux_corrected = flux_raw-flux_offset;
    flux_sum = 0; % Initialization
    for n_of_cycle = 2:Ncycles_round-1 % The first and last cycles are not include din the calculations
        initial_point_cycle = (n_of_cycle-1)*samples_per_cycle;
        final_point_cycle = initial_point_cycle+samples_per_cycle-1;
        flux_sum = flux_sum+max(flux_corrected(initial_point_cycle:final_point_cycle))-min(flux_corrected(initial_point_cycle:final_point_cycle)); % For every cycle, we add Bpk-pk
    end
    Flux(n)=flux_sum/(Ncycles_round-2)/2; % Divide the sum of the flux over the number of cycles used to calculate it, and over two to have the amplitude
    
    if n==round(Ndata/2) && display==1
        figure; hold on;
        plot(flux_raw*1e3, 'r');
        plot(flux_offset*1e3, 'b');
        plot(flux_corrected*1e3, 'k');
        plot([1, Nsamples], [Flux(n)*1e3, Flux(n)*1e3], '--k');
        plot([1, Nsamples], [-Flux(n)*1e3, -Flux(n)*1e3], '--k');
        plot([samples_per_cycle, samples_per_cycle], [-Flux(n)*1e3, Flux(n)*1e3], '--k');
        plot([final_point_cycle, final_point_cycle], [-Flux(n)*1e3, +Flux(n)*1e3], '--k');
        xlabel('Sample');
        ylabel('Flux density [mT]');
        legend('Raw','Offset','Corrected')
        title(['Datapoint=', num2str(n)]);
        drawnow();
    end

end

% Plot the AC flux amplitude
if display==1
    figure;
    plot(Flux*1e3, '.k');
    xlabel('Datapoint');
    ylabel('Flux density amplitude [mT]');
    drawnow();
end

end