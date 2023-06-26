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
        Hdc = Data.Hdc; % A/m'
        DutyP = Data.DutyP; % per unit
        DutyN = Data.DutyN; % per unit
        Freq = Data.Frequency; % Hz
        Flux = Data.Flux; % T
        Loss = Data.Volumetric_Loss; % W/m3
        B = Data.B_Field; % T
        H = Data.H_Field; % A/m

        Info_Setup = [Data.Date_info, ' --- ', Data.Place_info, ' --- ', Data.Trap_info, ' --- ', Data.Sine_info, ' --- ', Data.Bias_info, ' --- ', Data.Temp_info, ' --- ', Data.Meas_info, ' --- ', Data.Acquisition_info];
        Info_Core = [Data.Material, ' --- ', Data.Shape, ' --- Ae ', num2str(Data.Effective_Area), ' --- Ve ', num2str(Data.Effective_Volume), ' --- le ', num2str(Data.Effective_Length), ' --- CoreN ', num2str(Data.CoreN), ' --- N1 ', num2str(Data.Primary_Turns), ' --- N2 ', num2str(Data.Secondary_Turns), ' --- Dataset ', num2str(Data.Dataset)];
        Info_Processing = [Data.Discarding_info, ' --- ', Data.Freq_info, ' --- ', Data.Cycle_info, ' --- ', Data.Processing_info, ' --- ', Data.Date_processing];

        Ndata = length(B(:,1)); % Number of datapoints in the whole run
        Ncycle = length(B(1,:)); % Number of samples per datapoint
        disp(['Processed.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Ncycle), ' samples per datapoint loaded'])
    else
        disp('Cycle.mat file not been generated yet, quit execution')
        return
    end

%% Saving the data for the scalar-to-scalar NN training
DataScalar2Scalar = struct(...
    'Info_Setup', Info_Setup,...
    'Info_Core', Info_Core,...
    'Info_Processing', Info_Processing,...
    'Temperature', round(Temp),...
    'Hdc', round(Hdc,1),...
    'Duty_P', round(DutyP,1),...
    'Duty_N', round(DutyN,1),...
    'Frequency', round(Freq),...
    'Flux_Density', round(Flux,3),...
    'Volumetric_Loss', round(Loss,2));

Ndata = length(Temp); % Number of datapoints
JSON = jsonencode(DataScalar2Scalar);
fprintf(fopen([pwd, '\', Material, '\', Shape, '\Dataset', num2str(Dataset), '\', mat_name, '_Scalar2Scalar.json'], 'w'), JSON); fclose('all');
disp(['Scalar2Scalar.json file saved, with ', num2str(Ndata), ' datapoints'])

%% Saving the data for sequence-to-sequence NN training downsampled

Downsample_rate = 2^3;
B_Downsampled = B(:,1:Downsample_rate:end);
H_Downsampled = H(:,1:Downsample_rate:end);

DataSeq2Seq = struct(...
    'Info_Setup', Info_Setup,...
    'Info_Core', Info_Core,...
    'Info_Processing', Info_Processing,...
    'Temperature', round(Temp),...
    'Frequency', round(Freq),...
    'B_Field', round(B_Downsampled,5),...
    'H_Field', round(H_Downsampled,3));

JSON = jsonencode(DataSeq2Seq);
fprintf(fopen([pwd, '\', Material, '\', Shape, '\Dataset', num2str(Dataset), '\', mat_name, '_Seq2Seq.json'], 'w'), JSON); fclose('all');
disp(['Seq2Seq_Downsampled.json file saved, with ', num2str(Ndata), ' datapoints'])

%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
return

%% Plot the B-H loop
n_plot = round(Ndata/2);
Time = (0:Ncycle-1).*Data.Sampling_Time;
Time_Downsampled = Time(:,1:Downsample_rate:end);
figure;
subplot(1,3,1);
hold on;
plot(Time(n_plot,:)*1e6, B(n_plot,:)*1e3, 'k');
plot(Time_Downsampled(n_plot,:)*1e6, B_Downsampled(n_plot,:)*1e3, '.r');
xlabel('$t$ [us]');
ylabel('$B$ [mT]');
subplot(1,3,2);
hold on;
plot(Time(n_plot,:)*1e6, H(n_plot,:), 'k');
plot(Time_Downsampled(n_plot,:)*1e6, H_Downsampled(n_plot,:), '.r');
xlabel('$t$ [us]');
ylabel('$H$ [A/m]');
subplot(1,3,3);
hold on;
plot(H(n_plot,:), B(n_plot,:)*1e3, 'k');
plot(H_Downsampled(n_plot,:), B_Downsampled(n_plot,:)*1e3, '.r');
xlabel('$H$ [A/m]');
ylabel('$B$ [mT]');
sgtitle(['Datapoint=', num2str(n_plot)]);
drawnow();