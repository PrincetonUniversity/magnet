function inaccurate_vector = FindInaccurateData(volt, curr, v_min, i_min, accuracy, clipping_fraction, display)
%ReadRawDataSine Detect innacurate raw data:
%   volt - voltage matrix, single precision
%   curr - current matrix, single precision
%   v_min - minimum value for the peak voltage accepted
%   i_min - minimum value for the peak current accepted
%   accuracy - accuracy of the scope, offset above this value discarded
%   clipping_fraction - fraction of the samples accepted to be clipped
%   display - additional plots and messages
%   discard_vector - vector equal to 1 when the data is not accurate
%
%   Deleting innacurate data for various reasons:
%   1) Peak voltage or current below a minimum value
%   2) Average voltage or current above the accuracy of the scope
%   3) Power below the voltage times current accuracy of the scope
%   4) Clipped data

Ndata = length(volt(:,1));
Nsamples = length(volt(1,:));
inaccurate_vector = zeros(Ndata,1); % Initialization

for n = 1:Ndata
    sample_volt = volt(n,:); % Not to carry the (n,:) all around
    sample_curr = curr(n,:);
    volt_peak = max(abs(sample_volt));
    curr_peak = max(abs(sample_curr));
    volt_mean = abs(mean(sample_volt));
    curr_mean = abs(mean(sample_curr));
    loss = mean(sample_volt.*sample_curr);

    % Voltage and current below minimum
    if volt_peak<v_min
        inaccurate_vector(n) = 1;
        if display==1
            disp(['Test ', num2str(n), ' discarded: peak voltage below minimum value'])
        end
    end
    if curr_peak<i_min
        inaccurate_vector(n) = 1;
        if display==1
            disp(['Test ', num2str(n), ' discarded: peak current below minimum value'])
        end    
    end
    % Maximum average voltage and current accepted (based on the DC gain accuracy of the scope)
    if volt_mean>volt_peak*accuracy
        % Error underestimated because the maximum is always above the peak value
        inaccurate_vector(n) = 1;
        if display==1
            disp(['Test ', num2str(n), ' discarded: average voltage too high'])
        end
    end
    if curr_mean>curr_peak*accuracy
        inaccurate_vector(n) = 1;
        if display==1
            disp(['Test ', num2str(n), ' discarded: average current too high'])
        end    
    end
    % Minimum power (based on the DC gain accuracy of the scope)
    if loss<volt_peak*accuracy*curr_peak*accuracy
        % Error underestimated because the maximum is always above the peak value
        inaccurate_vector(n)=1;
        if display==1
            disp(['Test ',num2str(n),' discarded: losses too low'])
        end    
    end
    % Clipped data   
    % Works even when there are peaks above the saturation value
    % Calculate the signal derivative, then, find if there is a vector of 
    % zeros of a given lenght inside it.
    if sum(strfind(diff(sample_volt), zeros(1,Nsamples*clipping_fraction)))>0
        inaccurate_vector(n) = 1;
        disp(['Test ', num2str(n), ' discarded: voltage waveform clipped'])
        continue
    end
    if sum(strfind(diff(sample_curr), zeros(1,Nsamples*clipping_fraction)))>0
        inaccurate_vector(n) = 1;
        disp(['Test ', num2str(n), ' discarded: current waveform clipped'])
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
end

disp([num2str(sum(inaccurate_vector)), ' discarded out of ', num2str(Ndata), ', ', num2str(Ndata-sum(inaccurate_vector)), ' remaining'])

end