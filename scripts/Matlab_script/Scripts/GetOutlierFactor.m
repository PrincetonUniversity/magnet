function Outlier_Factor = GetOutlierFactor(Freq, Flux, Loss, Run, closeness_max, display)
%GetOutlierFactor computes the Outlier factor based on lsqcurvefit (based on Evan's code)
%   Freq - frequency vector (Hz)
%   Flux - AC flux density amplitude vector (T)
%   Loss - Volumetric losses (W/m3)
%   Run - Identifier for each B and f sweep
%   closeness_max - How far in terms of log10(f) and log10(B) to check
%   display - additional plots and messages
%   outlier factor - percentage vector 
%
%   Finds the outlier factor for each datapoint. The outlier factor is a
%   measure of how far the measured losses are from the expected value,
%   which is obtained from the neighbor datapoints. Basically, the losses
%   of the datapoints nearby are weighted based on how far are from the
%   selected datapoint, and used to extract the local SE parameters.
%   The Outlier Factor is defined as 1 minus this expected value divided over the
%   losses of each specific datapoint.
%   This outlier function defines "closseness" as 1 as the same point and 
%   as 0 as up "Closeness" decades from the point in either directions 
%   (for Closeness=1 it means times 10 or over 10 in f or B, or 5.09 times 
%   the distance in B and f at the same time)
%   It is log in the middle (this funtion has a conical shape in the loglog 
%   plot from 1 to 0), anything below 0 is left as 0


Ndata = length(Freq(:));
Outlier_Factor = zeros(Ndata,1); % Initialization
% Configuration of the SE optimization
%options = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','FunctionTolerance',1.0e-10,'MaxIterations',500);
opts = optimset('Display','off');
parameters_0 = [0.1,1.5,2.5]; % Initial conditions for k, alpha and beta
parameters_min = [1e-8,0.0,1.0]; % Min values
parameters_max = [1e8,4.0,4.0]; % Max values

for n = min(Run):max(Run)

    Freq_run = Freq(Run==n); Flux_run = Flux(Run==n); Loss_run = Loss(Run==n); % Read the data
    Ndata_run = length(Loss_run);
    if Ndata_run==0
        disp(['There is no data in the run ', num2str(n), ', please check'])
        continue
    end

    Loss_localSE_run = zeros(Ndata_run,1); % Initialization
    k_run = zeros(Ndata_run,1); alpha_run = zeros(Ndata_run,1); beta_run = zeros(Ndata_run,1);

    for i = 1:Ndata_run
        Closeness_run = zeros(Ndata_run,1); % Initialization
        for j = 1:Ndata_run
          Closeness_run(j) = 1-sqrt(log10(Freq_run(i)/Freq_run(j))^2+log10(Flux_run(i)/Flux_run(j))^2)/closeness_max;
        end
        Closeness_run(Closeness_run<0) = 0; Closeness_run(i) = 0; % Avoid having negative distances and itself not inlcuded

        % SE optimization weighted by closeness.
        parameters = lsqcurvefit(@WeightedLog10SE, parameters_0, [Freq_run Flux_run Closeness_run], (Closeness_run.*log10(Loss_run)), parameters_min, parameters_max, opts);
        parameters_0 = [parameters(1), parameters(2), parameters(3)]; % to speed up convergence, the initial k, alpha and beta are used as the initial point for the next point
    
        if sum((parameters_min>=parameters)+(parameters>=parameters_max))>0
            disp(['Careful, one of the Steinmetz parameters has saturated to the max or min value, [k, alpha, beta]=', num2str(parameters)])
        end

        k_run(i) = parameters(1);
        alpha_run(i) = parameters(2);
        beta_run(i) = parameters(3);

        Loss_localSE_run(i) = k_run(i)*Freq_run(i)^alpha_run(i)*Flux_run(i)^beta_run(i);
    
        if i==round(Ndata_run/2) && n==round(max(Run)/2) && display==1
        figure;
        scatter3(Flux_run*1e3, Freq_run/1e3, Closeness_run, 15, Closeness_run, 'filled');
        c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
        c.Label.String = "Weight";
        xlabel('AC flux density amplitude [mT]');
        ylabel('Frequency [kHz]');
        zlabel('Weight');
        title(['Run = ',num2str(n), ', Datapoint = ', num2str(i)]);
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); view(2);
        drawnow();
        end
    end
    
    if display==1 && n==round(max(Run)/2) % For debugging
    figure
    subplot(1,3,1);
    scatter3(Flux_run*1e3, Freq_run/1e3, k_run, 15, k_run, 'filled');
    c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
    c.Label.String = "$k$";  
    xlabel('$B_{pk}$ [mT]');
    ylabel('$f$ [kHz]');
    zlabel('$k$');
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
    subplot(1,3,2);
    scatter3(Flux_run*1e3, Freq_run/1e3, alpha_run, 15, alpha_run,'filled');
    c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter =' latex';
    c.Label.String = "$\alpha$";
    xlabel('$B_{pk}$ [mT]');
    ylabel('$f$ [kHz]');
    zlabel('$\alpha$');
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); view(2);
    subplot(1,3,3);
    scatter3(Flux_run*1e3, Freq_run/1e3, beta_run, 15, beta_run, 'filled');hold on
    c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter ='latex';
    c.Label.String = "$\beta$";
    xlabel('$B_{pk}$ [mT]');
    ylabel('$f$ [kHz]');
    zlabel('$\beta$');
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); view(2);
    sgtitle(['Run = ', num2str(n)]);
    drawnow();
    end

    Outlier_Factor_run = (Loss_localSE_run-Loss_run)./Loss_run*100; % It is a percentage
    if display==1
        disp(['Run ', num2str(n), ': the average outlier factor is ', num2str(round(mean(abs(Outlier_Factor_run)),2)), ' % and the peak outlier factor is ', num2str(round(max(abs(Outlier_Factor_run)),1)), ' %'])
    end

    Outlier_Factor(Run==n) = Outlier_Factor_run;    
end

if display==1
    figure;
    plot(Outlier_Factor, '.k');
    xlabel('Datapoint');
    ylabel('Outlier factor [\%]');
    drawnow();
end
disp(['The average outlier factor is ', num2str(round(mean(abs(Outlier_Factor)),2)), ' % and the peak outlier factor is ', num2str(round(max(abs(Outlier_Factor)),1)), ' %'])

end
