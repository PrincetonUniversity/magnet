% To generate the single cycle files
% The input required is the combined.mat file and the Test_Info.xlsx.
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
%Material = 'N30'; Shape = '22.1X13.7X6.35'; Dataset = 1;
%Material = 'N27'; Shape = 'R20.0X10.0X7.0'; Dataset = 1;
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

%% Read the combined.mat file
    if isfile([path_root, mat_name, '_Combined.mat'])
        load([path_root, mat_name, '_Combined.mat']); % Read the .mat file
        
        Volt = Data.Voltage; % V
        Curr = Data.Current; % A
        Ts = Data.Sampling_Time; % s
        Temp_command =  Data.Temperature_command; % C
        Hdc_command = Data.Hdc_command; % A/m
        DutyP_command = Data.DutyP_command; % per unit
        DutyN_command = Data.DutyN_command; % per unit
        Freq_command = Data.Frequency_command; % Hz
        Flux_command = Data.Flux_command; % T
        Discarding_info = Data.Discarding_info;
        Date_processing = Data.Date_processing; 

        Ndata = length(Volt(:,1)); % Number of datapoints in the whole run
        Nsamples = length(Volt(1,:)); % Number of samples per datapoint
        disp(['Combined.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Nsamples), ' samples per datapoint loaded'])
    else
        disp('Combined.mat file not been generated yet, quit execution')
        return
    end

%% Obtain the frequency. Constant time step considered only.
    % Accurate frequency identification using Power Spectral Density for
    % the voltage sequence
    f_resolution = 10; %10; % In Hz
    f_distance = 2e3; %12.5e3/2;
    Freq = GetFreqPWELCH(Volt, Ts, Freq_command, f_resolution, f_distance, display); % 100 to 1000 times faster than the previous method    

% Max error in frequency
    [~,WC_idx] = max(abs(Freq-Freq_command)./Freq_command);
    GetFreqPWELCH(Volt(WC_idx,:), Ts(WC_idx), Freq_command(WC_idx), f_resolution, f_distance, display);close
    sgtitle(['Worst case: ', num2str(WC_idx)])
    
%% Code for the single cycle
    % Averaging the Volt file into a single cycle, only valid for constant timesteps, two cycles per waveform minimum!
    Ncycle = 1024;%128; % Number of points per cycle for the single-cycle waveform

%% Voltage
    [Volt_Processed, NRMSE_Volt] = GetSingleCycle(Volt, Freq, Ts, Ncycle, display, 1);
    disp('Voltage single-cycle algorithm done')
    
%% Current
    [Curr_Processed, NRMSE_Curr] = GetSingleCycle(Curr, Freq, Ts, Ncycle, display, 1);
    disp('Current single-cycle algorithm done')

%% Max error
    for display_aux=1:display
        [~,WC_idx] = max(NRMSE_Volt);
        GetSingleCycle(Volt(WC_idx,:), Freq(WC_idx), Ts(WC_idx), Ncycle, display, 0); % Plot the worst case
        sgtitle(['Worst case datapoint=', num2str(WC_idx), '; NRMSE=', num2str(round(max(NRMSE_Volt)*100,2)), '\%'])
        [~,WC_idx] = max(NRMSE_Curr);
        GetSingleCycle(Curr(WC_idx,:), Freq(WC_idx), Ts(WC_idx), Ncycle, display, 0); % Plot the worst case
        sgtitle(['Worst case datapoint=', num2str(WC_idx), '; NRMSE=', num2str(round(max(NRMSE_Curr)*100,2)), '\%'])
    end

%% Shifting the starting point to the maximum voltage
    % Shifting the cycle so it start at the maximum voltage
    [~,Nstart] = max(Volt_Processed,[],2);
    Volt_Processed_Double = [Volt_Processed, Volt_Processed];
    Curr_Processed_Double = [Curr_Processed, Curr_Processed];
    Volt_Cycle = zeros(Ndata,Ncycle); Curr_Cycle = zeros(Ndata,Ncycle);
    for n = 1:Ndata
        Volt_Cycle(n,:) = Volt_Processed_Double(n,Nstart(n):Nstart(n)+Ncycle-1); % Displaced so the waveform starts where it is supposed to start
        Curr_Cycle(n,:) = Curr_Processed_Double(n,Nstart(n):Nstart(n)+Ncycle-1);   
    end
%% Removing the average voltage
    Volt_Cycle = Volt_Cycle - mean(Volt_Cycle,2); % Remove the average

%% Sampling time
    Ts_Cycle = 1./Freq/Ncycle; % Sampling time

%% Finding points with very high NRMSE or where the frequency does not match
    f_dist_check = f_distance*3/4;
    disp([num2str(sum(abs(Freq-Freq_command)>=f_dist_check)), ' discarded out of ', num2str(Ndata), ': Frequency too far'])
    NRMSE_max = 10/100; % Maximum normalized RMS error for the sequences
    disp([num2str(sum((NRMSE_Volt>=NRMSE_max)>0)), ' discarded out of ', num2str(Ndata), ': NRMSE Voltage'])
    disp([num2str(sum((NRMSE_Curr>=NRMSE_max)>0)), ' discarded out of ', num2str(Ndata), ': NRMSE Current'])
    discarded =(abs(Freq-Freq_command)>f_dist_check)|(NRMSE_Volt>NRMSE_max)|(NRMSE_Curr>NRMSE_max);

    figure;
    hold on;
    s = scatter3(Flux_command*1e3, Freq_command*1e-3, Hdc_command, 15, discarded, 'filled');
    s.MarkerFaceAlpha = 0.5; 
    c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
    c.Label.String = 'Discarded';
    xlabel('$B_{ac}$ [mT]');
    ylabel('$f$ [kHz]');
    zlabel('$H_{dc}$~[A/m]');
    xticks([10 20 30 50 100 200 300])
    yticks([50 100 200 300 500])
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); view(3);

%% Discarding those points
    Volt_Cycle(discarded,:) = [];
    Curr_Cycle(discarded,:) = [];
    Ts_Cycle(discarded,:) = [];
    Temp_command(discarded) = [];
    Hdc_command(discarded) = [];
    DutyP_command(discarded) = [];
    DutyN_command(discarded) = [];
    Freq_command(discarded) = [];
    Flux_command(discarded) = [];

    Ndata = length(Volt_Cycle(:,1));

%% Saving the single cycle data as .mat with the outlier factor
    Freq_info = ['Frequency estimation based on the power spectral density estimate via Welchs method using the voltage sequence with a frequency resolution of ', num2str(f_resolution), ' Hz for a range of ', num2str(f_distance), ' Hz around the commanded frequency; datapoints are discarded if the frequency is more than ', num2str(f_dist_check), ' Hz away from the commanded value']; %FFT or input frequency'];
    Cycle_info = ['Saving a single cycle by averaging all switching cycles but the last one if incomplete based on the frequency algorithm; using cubic spline interpolation with ', num2str(Ncycle), ' points for the waveforms; the average voltage is removed; datapoints an averaged normalized RMS error of all cycles of the voltage/current with respect to the averaged voltage/current from all cycles above ', num2str(NRMSE_max*100) ,' percent are removed'];
    Data = struct(...
        'Date_info', Date_info, 'Place_info', Place_info, 'Trap_info', Trap_info, 'Sine_info', Sine_info, 'Bias_info', Bias_info, 'Temp_info', Temp_info, 'Meas_info', Meas_info, 'Acquisition_info', Acquisition_info,...
        'Discarding_info', Discarding_info, 'Freq_info', Freq_info, 'Cycle_info', Cycle_info, 'Date_processing', Date_processing,...
        'Material', Material, 'Shape', Shape, 'Effective_Area', Ae, 'Effective_Volume', Ve, 'Effective_Length', Le,...
        'CoreN', CoreN, 'Primary_Turns', N1, 'Secondary_Turns', N2, 'Dataset', Dataset,...
        'Voltage', Volt_Cycle, 'Current', Curr_Cycle, 'Sampling_Time', Ts_Cycle,...
        'Temperature_command', Temp_command,'Hdc_command', Hdc_command, 'DutyP_command', DutyP_command, 'DutyN_command', DutyN_command, 'Frequency_command', Freq_command, 'Flux_command', Flux_command);

    save([path_root, mat_name, '_Cycle.mat'], 'Data', '-v7.3');
    disp(['Cycle.mat file saved ', num2str(Ndata), ' datapoints with ', num2str(Ncycle), ' samples per datapoint'])

%% End write
end
if write_or_read==0

%% Read the cycle .mat file
    if isfile([path_root, mat_name, '_Cycle.mat'])
        load([path_root, mat_name, '_Cycle.mat']); % Read the .mat file
        
        Volt_Cycle = Data.Voltage; % V
        Curr_Cycle = Data.Current; % A
        Ts_Cycle = Data.Sampling_Time; % s
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

        Ndata = length(Volt_Cycle(:,1)); % Number of datapoints in the whole run
        Ncycle = length(Volt_Cycle(1,:)); % Number of samples per datapoint
        disp(['Cycle.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Ncycle), ' samples per datapoint loaded'])
    else
        disp('Cycle.mat file not been generated yet, quit execution')
        return
    end

%% End read
end

%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
