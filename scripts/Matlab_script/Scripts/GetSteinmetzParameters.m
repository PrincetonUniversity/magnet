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
    
    parameters = lsqcurvefit(@WeightedLog10SE, parameters_0, [Freq Flux ones(length(Freq),1)], log10(Loss), parameters_min, parameters_max, opts);
    % ones(length(Freq),1) as the weight is equal for all points
    disp(['Steinmetz parameters: k=', num2str(parameters(1)), ', alpha=', num2str(parameters(2)), ', beta=', num2str(parameters(3))])
     
    Loss_SE = 10.^WeightedLog10SE(parameters, [Freq Flux ones(length(Freq),1)]);
    mismatch_loss = (Loss-Loss_SE)./Loss; % Error with this method

    k = parameters(1);
    alpha = parameters(2);
    beta = parameters(3);

    if sum((parameters_min>parameters)+(parameters>parameters_max))>0
        disp('Careful, one of the Steinmetz parameters has saturated to the max or min value')
    end

    if display==1
        disp(['The average mismatch with the SE parameters is ',num2str(mean(abs(mismatch_loss*100))),' % and the peak mismatch is ',num2str(max(abs(mismatch_loss*100))),' %'])
        figure;
        subplot(1,2,1)
        scatter3(Flux*1e3, Freq/1e3, Loss/1e3, 15, mismatch_loss*100, 'filled');
        c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
        c.Label.String = 'Mismatch in losses [\%]';
        xlabel('AC flux density amplitude [mT]');
        ylabel('Frequency [kHz]');
        zlabel('Loss density [kW/m$^3$]');
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
        drawnow();
        [mismatch_sorted, index_sorted]=sort(mismatch_loss);
        subplot(1,2,2)
        plot3(index_sorted, 1:Ndata, sort(mismatch_sorted)*100, '.k');
        xlabel('Datapoint');
        ylabel('Datapoint (sorted)');
        zlabel('Mismatch in losses [\%]');
        view(90,0);
        sgtitle(['$k = $', num2str(k),', $\alpha = $', num2str(alpha),', $\beta = $', num2str(beta), ' (Hz, T and W/m$^3$)']);
        drawnow();
    end
end