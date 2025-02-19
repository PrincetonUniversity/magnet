function f = GetFreqPWELCH(sequences, Ts, f_init, f_res, f_dist, display)
%GetFreqPWELCH Obtain the frequency near a prefedined frequency using the
%pwelch function, from Thomas' code.
%   sequences - either the voltage or the current matrix
%   Ts - sampling time vector
%   f_init - frequency to search around
%   f_res - desired frequency resolution
%   f_dist - maximum distance to the initial frequecy to look for
%   display - additional plots and messages
%   f - interpolated frequency vector

Ndata = length(sequences(:,1));
Nsamples = length(sequences(1,:));
f = zeros(Ndata,1); % Initialization

for n = 1:Ndata

    ts = Ts(n); 
    sample = sequences(n,:);
    f_ref= f_init(n);

    % frequency to be checked
    f_vec = round(max(f_ref-f_dist,0)/f_res)*f_res:f_res:round((f_ref+f_dist)/f_res)*f_res;

    % compute the power spectrum
    P_vec = pwelch(sample, gausswin(Nsamples), [], f_vec, 1/ts);
    
    % power spectrum has a maximum at the desired frequency
    [~, idx] = max(P_vec);
    f(n) = f_vec(idx);

    if display==1 && rem(n,1000)==0  % Every 100 calculations
        disp(['Frequency algorithm: ', num2str(round(n/Ndata*100,1)), '% done']);
    end    

    if n==round(Ndata/2) && display==1
        figure;
        subplot(1,2,1);
        plot((1:Nsamples)*ts*1e6, sample, 'k'); % Only an extra sample for the plot
        xlabel('Time [us]');
        ylabel('Amplitude');
        subplot(1,2,2); hold on;
        plot(f_vec/1e3, P_vec, '.k');
        plot(f_vec(idx)/1e3, P_vec(idx), 'ok');
        plot([f_ref f_ref]/1e3, [max(P_vec) min(P_vec)] , '--r');
        xlabel('Frequencies [kHz]');
        ylabel('Amplitude');
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
        sgtitle(['Datapoint=', num2str(n)]);
        drawnow();
    end
end
if display==1
    figure;
    plot(f*1e-3, '.k');
    xlabel('Datapoint');
    ylabel('$f$ [kHz]');
    drawnow();
end
end
