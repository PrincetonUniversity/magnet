% To generate the NN training files
% The input required is the processed.mat file and the Test_Info.xlsx.
% The output is saved in the dataset folder
% Contact: Diego Serrano, ds9056@princeton.edu, Princeton University

%% Clear previous varaibles and add the paths
clear % Clear variable in the workspace
clc % Clear command window
close all % Close all open figures
addpath('Scripts') % Add the folder where the scripts are located
cd ..; % To go to the previous folder 

path_webpage = 'C:\Users\diego\Desktop\data_processing_files_and_example\';  % IMPORTANT!! Relative paths won't work so change it in your computer

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

%% Read the processed .mat file
if isfile([path_root, mat_name, '_Processed.mat'])
    load([path_root, mat_name, '_Processed.mat']); % Read the .mat file
    
    Temp = Data.Temperature; % C
    Hdc = Data.Hdc; % A/m
    DutyP = Data.DutyP; % per unit
    DutyN = Data.DutyN; % per unit
    Freq = Data.Frequency; % Hz
    Flux = Data.Flux; % T
    Loss = Data.Volumetric_Loss; % W/m3

    Info_Setup = [Data.Date_info, ' --- ', Data.Place_info, ' --- ', Data.Trap_info, ' --- ', Data.Sine_info, ' --- ', Data.Bias_info, ' --- ', Data.Temp_info, ' --- ', Data.Meas_info, ' --- ', Data.Acquisition_info];
    Info_Core = [Data.Material, ' --- ', Data.Shape, ' --- Ae ', num2str(Data.Effective_Area), ' --- Ve ', num2str(Data.Effective_Volume), ' --- le ', num2str(Data.Effective_Length), ' --- CoreN ', num2str(Data.CoreN), ' --- N1 ', num2str(Data.Primary_Turns), ' --- N2 ', num2str(Data.Secondary_Turns), ' --- Dataset ', num2str(Data.Dataset)];
    Info_Processing = [Data.Discarding_info, ' --- ', Data.Freq_info, ' --- ', Data.Cycle_info, ' --- ', Data.Processing_info, ' --- ', Data.Date_processing];

    disp(['Processed.mat file loaded, ', num2str(length(Loss)), ' datapoints'])
else
    disp('Cycle.mat file not been generated yet, quit execution')
    return
end

%% Saving the JSON file for the webpage
DutyP(isnan(DutyP)) = -1;
DutyN(isnan(DutyN)) = -1;

DataWebpageDatabase = struct(...
    'Material', Material,...
    'Info_Setup', Info_Setup,...
    'Info_Core', Info_Core,...
    'Info_Processing', Info_Processing,...
    'Frequency', single(round(Freq)),...
    'Flux_Density', single(round(Flux,4)),...
    'DC_Bias', single(round(Hdc)),...
    'Duty_P', single(round(DutyP,1)),...
    'Duty_N', single(round(DutyN,1)),...
    'Temperature', single(round(Temp)),...
    'Power_Loss' ,single(round(Loss,2)));

path_webpage = [path_webpage, '\', Material, '\', Shape, '\Dataset', num2str(Dataset), '\',];

JSON = jsonencode(DataWebpageDatabase);
fprintf(fopen([path_webpage, Material, '_database.json'], 'w'), JSON); fclose('all');
disp(['Webpage .json file saved, with ', num2str(length(Loss)), ' datapoints'])

%% Getting the updated Steinmetz params at 25 C and 0 dc bias
subset = (DutyP==-1)&(DutyN==-1)&(Temp==25)&(round(Hdc/5)*5==0);
[k, alpha, beta] = GetSteinmetzParameters(Freq(subset), Flux(subset), Loss(subset), 1);
% Calculating the iGSE parameters
theta_vect = 0:0.0001:2*pi;
theta_int = cumtrapz(theta_vect, abs(cos(theta_vect)).^alpha.*2^(beta-alpha));
k_iGSE = k/((2*pi)^(alpha-1)*theta_int(end));
disp(['iGSE parameters: ki=', num2str(k_iGSE), ', alpha=', num2str(alpha),', beta=', num2str(beta)])

%% Create the .zip file with the .txt info file and the .cvs single cycle waveforms
% Save the .txt file
fid = fopen([path_webpage, 'test_info.txt'], 'w'); % Clear previous content
fprintf(fid, 'Core information: '); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Material: '); fprintf(fid, Data.Material); fprintf(fid, '\n');
fprintf(fid, 'Shape: '); fprintf(fid, Data.Shape); fprintf(fid, '\n');
fprintf(fid, 'Core number (when several cores are tested): '); fprintf(fid, num2str(Data.CoreN)); fprintf(fid, '\n');
fprintf(fid, 'Dataset (when the same core is measured several times): '); fprintf(fid, num2str(Data.Dataset)); fprintf(fid, '\n');
fprintf(fid, 'Primary turns: '); fprintf(fid, num2str(Data.Primary_Turns)); fprintf(fid, '\n');
fprintf(fid, 'Secondary turns: '); fprintf(fid, num2str(Data.Secondary_Turns)); fprintf(fid, '\n');
fprintf(fid, 'Effective length (m): '); fprintf(fid, num2str(Data.Effective_Length)); fprintf(fid, '\n');
fprintf(fid, 'Effective area (m2): '); fprintf(fid, num2str(Data.Effective_Area)); fprintf(fid, '\n');
fprintf(fid, 'Effective volume (m3): '); fprintf(fid, num2str(Data.Effective_Volume)); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Setup information: '); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Sinusoidal excitation: '); fprintf(fid, Data.Sine_info); fprintf(fid, '\n');
fprintf(fid, 'PWM excitation: '); fprintf(fid, Data.Trap_info); fprintf(fid, '\n');
fprintf(fid, 'DC bias: '); fprintf(fid, Data.Bias_info); fprintf(fid, '\n');
fprintf(fid, 'Temperature control: '); fprintf(fid, Data.Meas_info); fprintf(fid, '\n');
fprintf(fid, 'Measurements: '); fprintf(fid, Data.Temp_info); fprintf(fid, '\n');
fprintf(fid, 'Acquisition: '); fprintf(fid, Data.Acquisition_info); fprintf(fid, '\n');
fprintf(fid, 'Place of the measurements: '); fprintf(fid, Data.Place_info); fprintf(fid, '\n');
fprintf(fid, 'Date of the measurements: '); fprintf(fid, Data.Date_info); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Processing information: '); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, Data.Discarding_info); fprintf(fid, '\n');
fprintf(fid, Data.Freq_info); fprintf(fid, '\n');
fprintf(fid, Data.Cycle_info); fprintf(fid, '\n');
fprintf(fid, 'Obtaining the rest of the variables: '); fprintf(fid, Data.Processing_info); fprintf(fid, '\n');
fprintf(fid, 'Post-processing date: '); fprintf(fid, Data.Date_processing);
fclose('all'); % Close this files (and all other files)

B_Cycle = Data.B_Field; % T
H_Cycle = Data.H_Field; % A/m
Ts_Cycle = 1./Data.Frequency/width(B_Cycle); % s
Temp_Cycle = Data.Temperature; % C

% Save the .csv files
writematrix(B_Cycle, [path_webpage, 'b_cycle.csv']);
writematrix(H_Cycle, [path_webpage, 'h_cycle.csv']);
writematrix(Ts_Cycle, [path_webpage, 'sampling_time.csv']);
writematrix(Temp_Cycle, [path_webpage, 'temperature.csv']);

zip([path_webpage, Material, '_cycle'], {...
    [path_webpage, 'b_cycle.csv'],...
    [path_webpage, 'h_cycle.csv'],...
    [path_webpage, 'sampling_time.csv'],...
    [path_webpage, 'temperature.csv'],...
    [path_webpage, 'test_info.txt']});
delete([path_webpage, 'b_cycle.csv']);
delete([path_webpage, 'h_cycle.csv']);
delete([path_webpage, 'sampling_time.csv']);
delete([path_webpage, 'temperature.csv']);
delete([path_webpage, 'test_info.txt']);
disp([Material, '_cycle.zip file generated, .csv and .txt files deleted']);

%% Create the .zip file with the info and raw voltage and current waveform
% Save the .txt file
fid = fopen([path_webpage, 'test_info.txt'], 'w'); % Clear previous content
fprintf(fid, 'Core information: '); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Material: '); fprintf(fid, Data.Material); fprintf(fid, '\n');
fprintf(fid, 'Shape: '); fprintf(fid, Data.Shape); fprintf(fid, '\n');
fprintf(fid, 'Core number (when several cores are tested): '); fprintf(fid, num2str(Data.CoreN)); fprintf(fid, '\n');
fprintf(fid, 'Dataset (when the same core is measured several times): '); fprintf(fid, num2str(Data.Dataset)); fprintf(fid, '\n');
fprintf(fid, 'Primary turns: '); fprintf(fid, num2str(Data.Primary_Turns)); fprintf(fid, '\n');
fprintf(fid, 'Secondary turns: '); fprintf(fid, num2str(Data.Secondary_Turns)); fprintf(fid, '\n');
fprintf(fid, 'Effective length (m): '); fprintf(fid, num2str(Data.Effective_Length)); fprintf(fid, '\n');
fprintf(fid, 'Effective area (m2): '); fprintf(fid, num2str(Data.Effective_Area)); fprintf(fid, '\n');
fprintf(fid, 'Effective volume (m3): '); fprintf(fid, num2str(Data.Effective_Volume)); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Setup information: '); fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, 'Sinusoidal excitation: '); fprintf(fid, Data.Sine_info); fprintf(fid, '\n');
fprintf(fid, 'PWM excitation: '); fprintf(fid, Data.Trap_info); fprintf(fid, '\n');
fprintf(fid, 'DC bias: '); fprintf(fid, Data.Bias_info); fprintf(fid, '\n');
fprintf(fid, 'Temperature control: '); fprintf(fid, Data.Meas_info); fprintf(fid, '\n');
fprintf(fid, 'Measurements: '); fprintf(fid, Data.Temp_info); fprintf(fid, '\n');
fprintf(fid, 'Acquisition: '); fprintf(fid, Data.Acquisition_info); fprintf(fid, '\n');
fprintf(fid, 'Place of the measurements: '); fprintf(fid, Data.Place_info); fprintf(fid, '\n');
fprintf(fid, 'Date of the measurements: '); fprintf(fid, Data.Date_info);
fclose('all'); % Close this files (and all other files)    

% Read the combined .mat file
if isfile([path_root, mat_name, '_Combined.mat'])
    load([path_root, mat_name, '_Combined.mat']); % Read the .mat file
    disp(['Combined.mat file loaded, ', num2str(length(Data.Voltage(:,1))), ' datapoints with ', num2str(length(Data.Voltage(1,:))), ' samples per datapoint loaded'])
else
    disp('Combined.mat file not been generated yet, quit execution')
    return
end

% Save the .csv files
writematrix(Data.Voltage, [path_webpage, 'voltage.csv']);
writematrix(Data.Current, [path_webpage, 'current.csv']);
writematrix(Data.Sampling_Time, [path_webpage, 'sampling_time.csv']);
writematrix(Data.Temperature_command, [path_webpage, 'temperature.csv']);

zip([path_webpage, Material, '_measurements'], {...
    [path_webpage, 'voltage.csv'],...
    [path_webpage, 'current.csv'],...
    [path_webpage, 'sampling_time.csv'],...
    [path_webpage, 'temperature.csv'],...
    [path_webpage, 'test_info.txt']});
delete([path_webpage, 'voltage.csv']);
delete([path_webpage, 'current.csv']);
delete([path_webpage, 'sampling_time.csv']);
delete([path_webpage, 'temperature.csv']);
delete([path_webpage, 'test_info.txt']);
disp([Material, '_measurements.zip file generated, .csv and .txt files deleted']);

%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
return