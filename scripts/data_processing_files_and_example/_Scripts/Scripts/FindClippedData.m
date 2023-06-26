function inaccurate_vector = FindClippedData(sequences, clipping_fraction, display)
%FindClippedData Detect innacurate raw data:
%   sequences - data matrix, single precision
%   clipping_fraction - maximum percentage of data clipped
%   display - additional plots and messages
%   inaccurate_vector - vector equal to 1 when the data is not accurate
%
%   Deleting clipped data

Ndata = length(sequences(:,1));
Nsamples = length(sequences(1,:));
inaccurate_vector = zeros(Ndata,1); % Initialization

for n = 1:Ndata
    sample = sequences(n,:); % Not to carry the (n,:) all around
    if sum(strfind(diff(sample), zeros(1,Nsamples*clipping_fraction)))>0
        inaccurate_vector(n) = 1;
        if display==1
            disp(['Test ', num2str(n), ' discarded: waveform clipped'])
        end
        continue
    end
end
% Plot inaccurate data
if display==1
    figure;
    plot(inaccurate_vector-1, '.k');
    xlabel('Datapoint');
    ylabel('Inaccurate (0=Yes)');
    axis([1 Ndata -0.5 0.5]);
    drawnow();
    disp([num2str(sum(inaccurate_vector)), ' discarded out of ', num2str(Ndata), ', ', num2str(Ndata-sum(inaccurate_vector)), ' remaining'])
end
end



