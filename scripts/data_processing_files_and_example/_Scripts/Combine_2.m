% This script combines the raw files into a single file for each material
% The input required are the raw mat files for each temperture and the 
% information in Test_Info.xlsx.
% The combined.mat will be automatically saved in the dataset folder
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

%% Data of the core, and test (MKS system used)
Date_processing = datestr(datetime('Today'), 'yyyy-mm-dd'); % Day when the output files are generated
[~,~, CoreN, N1, N2, Le, Ae, Ve, Date_info, Place_info, Trap_info, Sine_info, Bias_info, Temp_info, Meas_info, Acquisition_info] = ReadTestInfo([path_root, 'Test_Info.xlsx']);

%% Start write
if write_or_read==1
   
%% Identify the different temperatures, from https://www.mathworks.com/matlabcentral/answers/166629-is-there-any-way-to-list-all-folders-only-in-the-level-directly-below-a-selected-directory
    files = dir(path_root); dirFlags = [files.isdir]; subFolders = files(dirFlags); subFolderNames = {subFolders(3:end).name};
    if isempty(subFolderNames)
        disp('Wrong folder, or no data, quit execution');
        return
    end
    Temperatures = [];
    for k=1:length(subFolderNames)
        Temperatures = [Temperatures str2double(cell2mat(subFolderNames(k)))]; %#ok<AGROW> % Please find a better way to do this!
    end
    
%% Read the raw .mat files
    Volt = []; % V
    Curr = []; % A
    Time = []; % s
    Temp_command = []; % C
    Hdc_command = []; % A/m
    Duty0_command = []; % per unit
    DutyP_command = []; % per unit
    Freq_command = []; % Hz
    Flux_command = []; % T
    for Temperature_it=Temperatures
        if isfile([path_root, num2str(Temperature_it), '\', mat_name, '_T', num2str(Temperature_it), '_Raw.mat'])
            load([path_root, num2str(Temperature_it), '\', mat_name, '_T', num2str(Temperature_it), '_Raw.mat']); % Read the .mat file
            
            Volt = [Volt; Data.Voltage]; %#ok<AGROW> % V
            Curr = [Curr; Data.Current]; %#ok<AGROW> % A
            Time = [Time; Data.Time]; %#ok<AGROW> % s
            Temp_command =  [Temp_command; Data.Temp_command]; %#ok<AGROW> % C
            Hdc_command = [Hdc_command; Data.Hdc_command]; %#ok<AGROW> % A/m
            Duty0_command = [Duty0_command; Data.Duty0_command]; %#ok<AGROW> % per unit
            DutyP_command = [DutyP_command; Data.DutyP_command]; %#ok<AGROW> % per unit
            Freq_command = [Freq_command; Data.Frequency_command]; %#ok<AGROW> % Hz
            Flux_command = [Flux_command; Data.Flux_command]; %#ok<AGROW> % T

            Ndata = length(Volt(:,1)); % Number of datapoints in the whole run
            Nsamples = length(Volt(1,:)); % Number of samples per datapoint
            disp([num2str(Temperature_it), ' C Raw.mat file loaded, ', num2str(Ndata), ' datapoints so far'])
        else
            disp('Raw.mat file not been generated yet, quit execution')
            return
        end
    end

%% Eliminating inaccurate data (clipping, voltage or current too low, average voltage too high)
    % Peak voltage too low
    Vmin_absolute = 1; % Minimum peak voltage acceptable
    Vpk_below_lim = max(max(Volt,[],2),-min(Volt,[],2))<Vmin_absolute;
    disp([num2str(sum(Vpk_below_lim)), ' discarded out of ', num2str(Ndata), ', ', num2str(Ndata-sum(Vpk_below_lim)), ' remaining: Vpk too low'])

    % Peak current too low
    Imin_absolute = 10e-3; % Minimum peak current acceptable
    Ipk_below_lim = max(max(Curr,[],2),-min(Curr,[],2))<Imin_absolute;
    disp([num2str(sum(Ipk_below_lim)), ' discarded out of ', num2str(Ndata), ', ', num2str(Ndata-sum(Ipk_below_lim)), ' remaining: Ipk too low'])

    % Remove clipped data
    Clipping_fraction = 0.1/100;
    Vclipped = FindClippedData(Volt, Clipping_fraction, 0);
    Iclipped = FindClippedData(Curr, Clipping_fraction, 0);

    % Total of points deleted (discard_vector=0 -> data is ok)
    discarded = Vpk_below_lim|Ipk_below_lim|Vclipped|Iclipped;
    
    % Plot discarded data
    figure;hold on;
    plot(Vpk_below_lim*0.8, '.');
    plot(Ipk_below_lim*0.9, '.');
    plot(Vclipped*0.7, '.');
    plot(Iclipped*0.6, '.');
    xlabel('Datapoint');
    ylabel('Inaccurate');
    legend('Vpk low', 'Ipk low', 'Clip V', 'Clip I')
    axis([1 Ndata 0.5 1]);
    drawnow();
    disp([num2str(sum(discarded)), ' discarded out of ', num2str(Ndata), ', ', num2str(Ndata-sum(discarded)), ' remaining'])


%% Deleting the inaccurate data
    Volt(discarded,:) = [];
    Curr(discarded,:) = [];
    Time(discarded,:) = [];
    Temp_command(discarded) = [];
    Hdc_command(discarded) = [];
    Duty0_command(discarded) = [];
    DutyP_command(discarded) = [];
    Freq_command(discarded) = [];
    Flux_command(discarded) = [];

    Ndata = length(Volt(:,1));

%% Obtain the sampling time (only if constant for each waveform)
    Ts = mean(diff(Time'),1)';%GetSamplingTime(Time);

%% Correct the current with the scope 50ohm termination
% IMPORTANT NOTICE! REMOVE THIS SECTION OF THE CODE ONCE THE PYHON CODE
% INCLUDES THIS CORRECTION DIRECTLY!
    R_sense = 0.983;
    Curr = Curr*(R_sense)/((50*R_sense)/(50+R_sense)); % Just increasing the current by a 2%

%% Saving the .mat combined data without the time
    Discarding_info = ['Data discarded: Voltage below ', num2str(Vmin_absolute), ' V; Current below ', num2str(Imin_absolute), ' A; Clipped signals for more than ', num2str(Clipping_fraction*100), ' percent of the sample length in a row'];
    DutyN_command = 1-DutyP_command-2*Duty0_command;
    
    Data = struct(...
        'Date_info', Date_info, 'Place_info', Place_info, 'Trap_info', Trap_info, 'Sine_info', Sine_info, 'Bias_info', Bias_info, 'Temp_info', Temp_info, 'Meas_info', Meas_info, 'Acquisition_info', Acquisition_info,...
        'Discarding_info', Discarding_info, 'Date_processing', Date_processing,...
        'Material', Material, 'Shape', Shape, 'Effective_Area', Ae, 'Effective_Volume', Ve, 'Effective_Length', Le,...
        'CoreN', CoreN, 'Primary_Turns', N1, 'Secondary_Turns', N2,'Dataset', Dataset,...
        'Voltage', Volt, 'Current', Curr, 'Sampling_Time', Ts,...
        'Temperature_command', Temp_command, 'Hdc_command', Hdc_command, 'DutyP_command', DutyP_command, 'DutyN_command', DutyN_command, 'Frequency_command', Freq_command, 'Flux_command', Flux_command);

    save([path_root, mat_name, '_Combined.mat'], 'Data', '-v7.3');
    disp('Combined.mat file saved')

%% End write
end
if write_or_read==0

    %% Read the combined .mat file
    if isfile([path_root, mat_name, '_Combined.mat'])
        load([path_root, mat_name, '_Combined.mat']); % Read the .mat file
        
        Volt = Data.Voltage; % V
        Curr = Data.Current; % A
        Ts = Data.Sampling_Time; % s
        Temp_command =  Data.Temperature; % C
        Hdc_command = Data.Hdc; % A/m'
        DutyP_command = Data.DutyP; % per unit
        DutyN_command = Data.DutyN; % per unit
        Freq_command = Data.Frequency; % Hz
        Flux_command = Data.Flux; % T
        Discarding_info = Data.Discarding_info;
        Date_processing = Data.Date_processing; 
    
        Ndata = length(Volt(:,1)); % Number of datapoints in the whole run
        Nsamples = length(Volt(1,:)); % Number of samples per datapoint
        disp(['Combined.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Nsamples), ' samples per datapoint loaded'])
    else
        disp('Combined.mat file not been generated yet, quit execution')
        return
    end
%% End read
end

%% Analize the raw data
figure
subplot(2,3,1)
plot(Temp_command,'.k')
xlabel('Datapoint');
ylabel('$T$~[$^\circ$C]');
xlim([1 Ndata])
subplot(2,3,2)
hold on;
plot(Hdc_command,'.k')
xlabel('Datapoint');
ylabel('$H_{dc}$~[A/m]');
xlim([1 Ndata])
subplot(2,3,3)
hold on;
plot(DutyP_command,'.k')
xlabel('Datapoint');
ylabel('$D_P$~[\%]');
xlim([1 Ndata])
ylim([0 1])
subplot(2,3,4)
hold on;
plot(DutyN_command,'.k')
xlabel('Datapoint');
ylabel('$D_N$~[\%]');
xlim([1 Ndata])
ylim([0 1])
subplot(2,3,5)
hold on;
plot(Freq_command*1e-3,'.k')
xlabel('Datapoint');
ylabel('$f$~[kHz]');
xlim([1 Ndata])
set(gca, 'YScale', 'log');
subplot(2,3,6)
hold on;
plot(Flux_command*1e3,'.k')
xlabel('Datapoint');
ylabel('$B_{ac}$~[mT]');
set(gca, 'YScale', 'log');
xlim([1 Ndata])
sgtitle(['Parameters: ', name]);
set(gcf,'units','points','position',[200,100,800,500])
set(findall(gcf,'-property','FontSize'),'FontSize',12)


%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
