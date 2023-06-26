function [k, alpha, beta] = GetSteinmetzParameters(Freq, Flux, Loss, display)
%GetSteinmetzParameters computes the Steinmetz parameters based on
%lsqcurvefit (based on Haoran's code)
%   Freq - frequency vector (Hz)
%   Flux - AC flux density amplitude vector (T)
%   Loss - Volumetric losses (W/m3)
%   display - additional plots and messages
%   k - Steinmetz parameter k
%   alpha - Steinmetz parameter alpha
%   beta - Steinmetz parameter beta
%
%   Find the function that makes log10(Pv) closer to the measured value,
%   where Pv=k*f^alpha*B^beta

    Ndata = length(Freq(:)); % Number of datapoints in the whole run

    % options = optimoptions('lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'FunctionTolerance', 1.0e-10, 'MaxIterations', 500);
    opts = optimset('Display','off');
    
    parameters_0 = [1,1.5,2.5]; % Initial conditions for k, alpha and beta
    parameters_min = [1e-6,1,1.5];
    parameters_max = [1000,2.5,3.5];
    
    parameters = lsqcurvefit(@Log10SE, parameters_0, [Freq Flux], log10(Loss), parameters_min, parameters_max, opts);
    
    Loss_SE = 10.^Log10SE(parameters, [Freq Flux]);
    mismatch_loss = (Loss-Loss_SE)./Loss; % Error with this method

    k = parameters(1);
    alpha = parameters(2);
    beta = parameters(3);

    disp(['Steinmetz parameters: k=', num2str(k), ', alpha=', num2str(alpha), ', beta=', num2str(beta)])
  
    if sum((parameters_min>parameters)+(parameters>parameters_max))>0
        disp('Careful, one of the Steinmetz parameters has saturated to the max or min value')
    end

    if display==1
        disp(['The average mismatch with the SE parameters is ',num2str(mean(abs(mismatch_loss*100))),'% and the peak mismatch is ',num2str(max(abs(mismatch_loss*100))),'%'])
        figure;
        scatter3(Flux*1e3, Freq/1e3, Loss/1e3, 15, mismatch_loss*100, 'filled');
        c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
        c.Label.String = '$(P_{meas}-P_{SE})/P_{meas}$ [\%]';
        xlabel('AC flux density amplitude [mT]');
        ylabel('Frequency [kHz]');
        zlabel('Loss density [kW/m$^3$]');
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
        drawnow();
    end
end