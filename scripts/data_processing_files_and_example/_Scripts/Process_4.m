% To generate the processed single cycle files
% The input required is the cycle.mat file and the Test_Info.xlsx.
% The output is saved in the dataset folder
% Contact: Diego Serrano, ds9056@princeton.edu, Princeton University

%% Clear previous varaibles and add the paths
clear % Clear variable in the workspace
clc % Clear command window
close all % Close all open figures
addpath('Scripts') % Add the folder where the scripts are located
cd ..; % To go to the previous folder 

%%%%%%%%%%%%%%%%%%%%%%% PLEASE SELECT WHETHER TO WRITE OR READ THE MAT FILE
write_or_read = 1; % 1 to write 0 to read;

%%%%%%%%%%%%%%%%%%%%% PLEASE SELECT THE MATERIAL, SHAPE, DATASET TO ANALYZE
Material = 'N87'; Shape = 'R34.0X20.5X12.5'; Dataset = 1;
%Material = 'N49'; Shape = 'R16.0X9.6X6.3'; Dataset = 1;
%Material = 'N27'; Shape = 'R20.0X10.0X7.0'; Dataset = 1;
%Material = 'N30'; Shape = '22.1X13.7X6.35'; Dataset = 1;
%Material = '3F4'; Shape = 'E-32-6-20-R'; Dataset = 1;
%Material = '3C90'; Shape = 'TX-25-15-10'; Dataset = 1;
%Material = '3C94'; Shape = 'TX-20-10-7'; Dataset = 1;
%Material = '3E6'; Shape = 'TX-22-14-6.4'; Dataset = 1;
%Material = '77'; Shape = '0014'; Dataset = 1;
%Material = '78'; Shape = '0076'; Dataset = 1;

path_root = [pwd, '\', Material, '\', Shape, '\Dataset', num2str(Dataset), '\']; % Path of this file
name = [Material, ' - ', Shape, ' - Dataset ', num2str(Dataset)];
mat_name = [Material, '_', Shape, '_Data', num2str(Dataset)];

%% Set the style for the plots
PlotStyle; close;
display = 1; % 1 to plot additional figures and messages, 0 to avoid additional plots.

%% Data of the core, and test (MKS system used)
Date_processing = datestr(datetime('Today'), 'yyyy-mm-dd'); % Day when the output files are generated
[~,~, CoreN, N1, N2, Le, Ae, Ve, Date_info, Place_info, Trap_info, Sine_info, Bias_info, Temp_info, Meas_info, Acquisition_info] = ReadTestInfo([path_root, 'Test_Info.xlsx']);

%% Start write
if write_or_read==1

%% Read the cycle.mat file
    if isfile([path_root, mat_name, '_Cycle.mat'])
        load([path_root, mat_name, '_Cycle.mat']); % Read the .mat file
        
        Volt = Data.Voltage; % V
        Curr = Data.Current; % A
        Ts = Data.Sampling_Time; % s
        Temp_command =  Data.Temperature_command; % C
        Hdc_command = Data.Hdc_command; % A/m'
        DutyP_command = Data.DutyP_command; % per unit
        DutyN_command = Data.DutyN_command; % per unit
        Freq_command = Data.Frequency_command; % Hz
        Flux_command = Data.Flux_command; % T

        Date_processing = Data.Date_processing;
        Discarding_info = Data.Discarding_info;
        Freq_info = Data.Freq_info;
        Cycle_info = Data.Cycle_info;

        Ndata = length(Volt(:,1)); % Number of datapoints in the whole run
        Ncycle = length(Volt(1,:)); % Number of samples per datapoint
        disp(['Cycle.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Ncycle), ' samples per datapoint loaded'])
    else
        disp('Cycle.mat file not been generated yet, quit execution')
        return
    end

%% Obtain the frequency, llux density and field intensity
    Freq = 1./Ts/Ncycle;

    B_Raw = cumtrapz(Volt,2)/N2/Ae.*Ts;
    B = B_Raw-mean(B_Raw,2); 

    H = Curr*N1/Le;

    % Plot the B-H loop
    for display_aux=1:display
        n_plot=round(Ndata/2);
        figure;
        subplot(2,3,1);
        plot((0:Ncycle-1)*Ts(n_plot)*1e6, B(n_plot,:)*1e3, 'k');
        xlabel('$t$ [us]');
        ylabel('$B$ [mT]');
        subplot(2,3,4);
        plot((0:Ncycle-1)*Ts(n_plot)*1e6, H(n_plot,:), 'k');
        xlabel('$t$ [us]');
        ylabel('$H$ [A/m]');
        subplot(2,3,[2,3,5,6]);
        plot(H(n_plot,:), B(n_plot,:)*1e3, 'k');
        xlabel('Field strenght, $H$ [A/m]');
        ylabel('Flux density, $B$ [mT]');
        sgtitle(['Datapoint=', num2str(n_plot)]);
        drawnow();
    end
%% Postprocessed parameters

% Type of excitation
    Excitation = isnan(DutyP_command); % 1 for sinusoidal;

% Flux density
    Flux = (max(B,[],2)-min(B,[],2))/2;

% Power loss
    Loss = mean(Volt.*Curr,2)/Ve; % Obtain losses again and divide by the effective volume, this time with the averaged waveforms

% Hdc
    Hdc = mean(H,2);

% Duty cycle
    % For Trapezoidal excitation. d2=d4 (d0) only.
    DutyP = zeros(Ndata,1); DutyN = zeros(Ndata,1);
    DutyP(Excitation) = NaN; DutyN(Excitation) = NaN;

    Duty_Nthresholds = 100; % Rounding and the number of points will determine how accurately the duty detection works (assuming the noise is as large as DeltaV, three times 1/res should be fine)
    Duty_remove_fraction = 1/100; % Remove the max and min fraction of the signal to clean the switching noise
    if min(Excitation)==0
        [DutyP(~Excitation), DutyN(~Excitation)] = GetDutyCycles(Volt(~Excitation,:), Freq(~Excitation), Ts(~Excitation), Duty_Nthresholds, Duty_remove_fraction, 0);
    end

% Temperature, nothing to do for now
    Temp = Temp_command;

%% Compare commanded and processed values
for display_aux=1:display
    figure
    subplot(2,3,1)
    plot(Temp,'.k')
    xlabel('Datapoint');
    ylabel('$T$~[$^\circ$C]');
    xlim([1 Ndata])
    subplot(2,3,2)
    hold on;
    plot(Hdc_command,'.r')
    plot(Hdc,'.b')
    xlabel('Datapoint');
    ylabel('$H_{dc}$~[A/m]');
    xlim([1 Ndata])
    legend('Commanded', 'Calculated')
    subplot(2,3,3)
    hold on;
    plot(DutyP_command,'.r')
    plot(DutyP,'.b')
    xlabel('Datapoint');
    ylabel('$D_P$~[\%]');
    xlim([1 Ndata])
    ylim([0 1])
    legend('Commanded', 'Calculated')
    subplot(2,3,4)
    hold on;
    plot(DutyN_command,'.r')
    plot(DutyN,'.b')
    xlabel('Datapoint');
    ylabel('$D_N$~[\%]');
    xlim([1 Ndata])
    ylim([0 1])
    legend('Commanded', 'Calculated')
    subplot(2,3,5)
    hold on;
    plot(Freq_command*1e-3,'.r')
    plot(Freq*1e-3,'.b')
    xlabel('Datapoint');
    ylabel('$f$~[kHz]');
    xlim([1 Ndata])
    legend('Commanded', 'Calculated')
    subplot(2,3,6)
    hold on;
    plot(Flux_command*1e3,'.r')
    plot(Flux*1e3,'.b')
    xlabel('Datapoint');
    ylabel('$B_{ac}$~[mT]');
    xlim([1 Ndata])
    legend('Commanded', 'Calculated')
    sgtitle(['Parameters: ', name]);
    set(gcf,'units','points','position',[200,100,800,500])
    set(findall(gcf,'-property','FontSize'),'FontSize',12)
end

%% Remove low quality data
% Losses too low
    Loss_min_th = 1e-3; %  1mW
    Loss_Too_Low = mean(Volt.*Curr,2)<Loss_min_th;
    for display_aux=1:display
        figure; hold on;
        plot(mean(Volt.*Curr,2), '.r');
        Aux_var = mean(Volt.*Curr,2).*(Loss_Too_Low==1);
        Aux_var(Aux_var==0) = NaN;
        plot(Aux_var, 'ok');
        xlabel('Datapoint');
        ylabel('Core Loss [W]');
        set(gca, 'YScale', 'log');
        set(gcf,'units','points','position',[200,100,800,500])
        set(findall(gcf,'-property','FontSize'),'FontSize',12)
    end
    disp([num2str(sum(Loss_Too_Low>0)), ' discarded out of ', num2str(Ndata), ': Loss too low'])

% Low quality factor
    Q_max_th = 100;
    S = max(Volt,[],2).*(max(Curr,[],2)-min(Curr,[],2))/2;
    Q = S./mean(Volt.*Curr,2);
    Q_Too_High = Q>Q_max_th;
    for display_aux=1:display
        figure; 
        subplot(1,2,1)
        plot(S, '.r');
        xlabel('Datapoint');
        ylabel('$S=V_{max}\cdot (I_{max}-I_{min})/2$ [W]');
        subplot(1,2,2)
        hold on;
        plot(Q, '.r');
        Aux_var = Q.*(Q_Too_High==1);
        Aux_var(Aux_var==0) = NaN;
        plot(Aux_var, 'ok');
        xlabel('Datapoint');
        ylabel('Quality Factor: $Q=S/$Loss');
        set(gcf,'units','points','position',[200,100,800,500])
        set(findall(gcf,'-property','FontSize'),'FontSize',12)
    end
    disp([num2str(sum(Q_Too_High>0)), ' discarded out of ', num2str(Ndata), ': Q too high'])

% THD too high in sinusoidal data
    THD = zeros(Ndata,1);
    THD_Too_High = zeros(Ndata,1);
    THD_max_th = 5/100; % 5% THD considered too high
    if sum(Excitation)>0
    % THD for the sinusoidal voltage
        THD(Excitation) = GetTHD(Volt(Excitation,:), Ts(Excitation), 100e6, 0); % up to 100MHz
        THD_Too_High(Excitation) = THD(Excitation)>THD_max_th;
        for display_aux=1:display
            figure; hold on;
            plot(THD*100, '.r');
            Aux_var = THD.*(THD_Too_High==1);
            Aux_var(Aux_var==0) = NaN;
            plot(Aux_var*100, 'ok');
            xlabel('Datapoint');
            ylabel('Voltage THD [\%]');
            set(gcf,'units','points','position',[200,100,800,500])
            set(findall(gcf,'-property','FontSize'),'FontSize',12)
        end
    end
    disp([num2str(sum(THD_Too_High>0)), ' discarded out of ', num2str(Ndata), ': THD too high'])

% Plotting the location of the removed data
    discarded = Loss_Too_Low+Q_Too_High|THD_Too_High;
    for display_aux=1:display
        figure;
        hold on;
        s = scatter3(Flux*1e3, Freq*1e-3, Hdc, 15, discarded, 'filled');
        s.MarkerFaceAlpha = 0.5; 
        c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
        c.Label.String = 'Discarded';
        xlabel('$B_{ac}$ [mT]');
        ylabel('$f$ [kHz]');
        zlabel('$H_{dc}$~[A/m]');
        xticks([10 20 30 50 100 200 300])
        yticks([50 100 200 300 500])
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); view(3);
        set(findall(gcf,'-property','FontSize'),'FontSize',12)
    end

%% Discarding those points
    Volt(discarded,:) = [];
    Curr(discarded,:) = [];
    Ts(discarded,:) = [];
    Temp(discarded) = [];
    Hdc(discarded) = [];
    DutyP(discarded) = [];
    DutyN(discarded) = [];
    Freq(discarded) = [];
    Flux(discarded) = [];
    Loss(discarded) = [];
    B(discarded,:) = [];
    H(discarded,:) = [];

    Ndata = length(Volt(:,1));

%% Saving the processed .mat file
    Processing_info = ['Flux estimated based on the integral of the voltage (the average flux is removed as the initial value is not known); Loss density calculated as the average product of the voltage and current (average voltage removed beforehand) divided by the effective volume; Duty cycle estimated by dividing the peak to peak voltage in ', num2str(Duty_Nthresholds), ' equally spaced thresholds and checking the percentage of the sequence that is above this threshold; the max and min ', num2str(Duty_remove_fraction*100), ' percent of the waveform is removed first'];
    Discarding_info = [Discarding_info, '; with a voltage THD above ', num2str(THD_max_th*100), ' percent; With losses below ', num2str(Loss_min_th), ' W; With a quality factor above ', num2str(Q_max_th)];

    Data = struct(...
        'Date_info', Date_info, 'Place_info', Place_info, 'Trap_info', Trap_info, 'Sine_info', Sine_info, 'Bias_info', Bias_info, 'Temp_info', Temp_info, 'Meas_info', Meas_info, 'Acquisition_info', Acquisition_info,...
        'Discarding_info', Discarding_info, 'Freq_info', Freq_info, 'Cycle_info', Cycle_info, 'Processing_info', Processing_info, 'Date_processing', Date_processing,...
        'Material', Material, 'Shape', Shape, 'Effective_Area', Ae, 'Effective_Volume', Ve, 'Effective_Length', Le,...
        'CoreN', CoreN, 'Primary_Turns', N1, 'Secondary_Turns', N2, 'Dataset', Dataset,...
        'Voltage', Volt, 'Current', Curr, 'Sampling_Time', Ts,...
        'Temperature', Temp, 'Hdc', Hdc, 'DutyP', DutyP, 'DutyN', DutyN, 'Frequency', Freq, 'Flux', Flux,...
        'Volumetric_Loss', Loss, 'B_Field', B, 'H_Field', H);

    save([path_root, mat_name, '_Processed.mat'], 'Data', '-v7.3');
    disp('Processed.mat file saved')

%% End write
end
if write_or_read==0

%% Read the processed .mat file
    if isfile([path_root, mat_name, '_Processed.mat'])
        load([path_root, mat_name, '_Processed.mat']); % Read the .mat file
        
        Volt = Data.Voltage; % V
        Curr = Data.Current; % A
        Ts = Data.Sampling_Time; % s
        Temp =  Data.Temperature; % C
        Hdc = Data.Hdc; % A/m'
        DutyP = Data.DutyP; % per unit
        DutyN = Data.DutyN; % per unit
        Freq = Data.Frequency; % Hz
        Flux = Data.Flux; % T
        Loss = Data.Volumetric_Loss; % W/m3
        B = Data.B_Field; % T
        H = Data.H_Field; % A/m

        Date_processing = Data.Date_processing;
        Discarding_info = Data.Discarding_info;
        Freq_info = Data.Freq_info;
        Cycle_info = Data.Cycle_info;
        Processing_info = Data.Processing_info;

        Ndata = length(B(:,1)); % Number of datapoints in the whole run
        Ncycle = length(B(1,:)); % Number of samples per datapoint
        disp(['Processed.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Ncycle), ' samples per datapoint loaded'])
    else
        disp('Processed.mat file not been generated yet, quit execution')
        return
    end

%% End read
end

%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
