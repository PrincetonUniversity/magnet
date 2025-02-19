% To generate the raw.mat file combining all the collected data for one temperature
% The files required are:
%   1. The information of the core under test, please fill the second
%   column in Test_Info.xlsx.
%   2. The voltage, current, time, and info data from the python code, in
%   separated .csv files, a row per datapoint, named X_Data_Curr.csv,
%   X_Data_Time.csv, X_Data_Volt.csv and X_Parameters.csv.
% The outputs will be automatically saved in the folder with the data
% Contact: Diego Serrano, ds9056@princeton.edu, Princeton University

%% Clear previous varaibles and add the paths
clear % Clear variable in the workspace
clc % Clear command window
close all % Close all open figures
addpath('Scripts') % Add the folder where the scripts are located
cd ..; % Go to the previous folder 

%%%%%%%%%%%%%%%%%%%%%%% PLEASE SELECT WHETHER TO WRITE OR READ THE MAT FILE
write_or_read = 1; % 1 to write 0 to read;

%%%%%%%% PLEASE SELECT THE MATERIAL, SHAPE, DATASET AND TEMPERATURE TO READ
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

Temp_range = [25 50 70 90];

path_root = [pwd, '\', Material, '\', Shape, '\Dataset', num2str(Dataset), '\']; % Path of this file
name = [Material, ' - ', Shape, ' - Dataset ', num2str(Dataset)];
mat_name = [Material, '_', Shape, '_Data', num2str(Dataset)];

%% Set the style for the plots
PlotStyle; close;

%% Data of the core, and test (MKS system used)
Date_processing = datestr(datetime('Today'), 'yyyy-mm-dd'); % Day when the output files are generated
[~,~, CoreN, N1, N2, Le, Ae, Ve, Date_info, Place_info, Trap_info, Sine_info, Bias_info, Temp_info, Meas_info, Acquisition_info] = ReadTestInfo([path_root, 'Test_Info.xlsx']);

%%
for Temperature = Temp_range
if write_or_read==1
%% Read the .csv files
    [Volt, Curr, Time, Params] = ReadRawData([path_root, num2str(Temperature), '\']);
    if not(isempty(Volt))
        Ndata = length(Volt(:,1)); % Number of datapoints in the whole run
        Nsamples = length(Volt(1,:)); % Number of samples per datapoint
        disp([num2str(Ndata), ' datapoints with ', num2str(Nsamples), ' samples per datapoint loaded'])
        Hdc = Params(:,1);
        Duty0 = Params(:,2);
        DutyP = Params(:,3);
        Freq = Params(:,4);
        Flux = Params(:,5);
        Temp = Temperature*ones(Ndata,1); % Just making the temperature a vector
    
    %% Save the raw measurements as a .mat file
        Data = struct( ...
            'Date_info', Date_info, 'Place_info', Place_info, 'Trap_info', Trap_info, 'Sine_info', Sine_info, 'Bias_info', Bias_info, 'Temp_info', Temp_info, 'Meas_info', Meas_info, 'Acquisition_info', Acquisition_info, 'Date_processing', Date_processing,...
            'Material', Material, 'Shape', Shape, 'Effective_Area', Ae, 'Effective_Volume', Ve, 'Effective_Length', Le,...
            'CoreN', CoreN, 'Primary_Turns', N1, 'Secondary_Turns', N2, 'Dataset', Dataset,...
            'Voltage', Volt, 'Current', Curr, 'Time', Time,...
            'Temp_command', Temp,'Hdc_command', Hdc, 'Duty0_command', Duty0, 'DutyP_command', DutyP, 'Frequency_command', Freq, 'Flux_command', Flux);
        
        save([path_root, num2str(Temperature), '\', mat_name, '_T', num2str(Temperature), '_Raw.mat'], 'Data', '-v7.3');
        disp('Raw.mat file saved')
    end

%% End write
end
if write_or_read==0
%% Read the raw .mat file
    if isfile([path_root, num2str(Temperature), '\', mat_name, '_T', num2str(Temperature), '_Raw.mat'])
        load([path_root, num2str(Temperature), '\', mat_name, '_T', num2str(Temperature), '_Raw.mat']); % Read the .mat file
        
        Volt = Data.Voltage; % V
        Curr = Data.Current; % A
        Time = Data.Time; % s
        Temp = Data.Temp_command; % C
        Hdc = Data.Hdc_command; % A/m
        Duty0 = Data.Duty0_command; % per unit
        DutyP = Data.DutyP_command; % per unit
        Freq = Data.Frequency_command; % Hz
        Flux = Data.Flux_command; % T

        Ndata = length(Volt(:,1)); % Number of datapoints in the whole run
        Nsamples = length(Volt(1,:)); % Number of samples per datapoint
        disp(['Raw.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Nsamples), ' samples per datapoint loaded'])
    else
        disp('Raw.mat file not been generated yet, quit execution')
        return
    end
%% End read
end
end

%% Analize the raw data
figure
subplot(2,3,1)
plot(Temp,'.k')
xlabel('Datapoint');
ylabel('$T$~[$^\circ$C]');
xlim([1 Ndata])
subplot(2,3,2)
plot(Hdc,'.k')
xlabel('Datapoint');
ylabel('$H_{dc}$~[A/m]');
xlim([1 Ndata])
subplot(2,3,3)
plot(Duty0,'.k')
xlabel('Datapoint');
ylabel('$D_0$');
xlim([1 Ndata])
subplot(2,3,4)
plot(DutyP,'.k')
xlabel('Datapoint');
ylabel('$D_P$');
xlim([1 Ndata])
subplot(2,3,5)
plot(Freq*1e-3,'.k')
xlabel('Datapoint');
ylabel('$f$~[kHz]');
set(gca, 'YScale', 'log');
xlim([1 Ndata])
subplot(2,3,6)
plot(Flux*1e3,'.k')
xlabel('Datapoint');
ylabel('$B_{ac}$~[mT]');
set(gca, 'YScale', 'log');
xlim([1 Ndata])
sgtitle(['Commanded parameters: ', name, ' at ', num2str(Temperature), ' $^\circ$C']);
set(gcf,'units','points','position',[200,100,800,500])
set(findall(gcf,'-property','FontSize'),'FontSize',12)

%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
