% To generate all the output files simply run this script.
% The input required are:
%   1. The information of the core under test, please fill the second
%   column in Inputs\Core_Info.xlsx.
%   2. The voltage, current and time data measured by the scope, in
%   separated .csv files, a row per datapoint, named Data_Curr_X.csv,
%   Data_Time_X.csv and Data_Volt_X.csv.
%   3. The datapoints extracted from the datasheet, a list of f, B, Pv and
%   T from the different power losses plots.
% The outputs will be automatically saved in the Output folder
% Contact: Diego Serrano, ds9056@princeton.edu, Princeton University

%% Clear previous varaibles and add the scripts path
clear % Clear variable in the workspace
clc % Clear command window
close all % Close all open figures

addpath('Scripts')

%% Set the style for the plots
set(groot, 'defaultFigureColor', [1 1 1]); % White background for pictures
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
% It is not possible to set property defaults that apply to the colorbar label. 
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultColorbarFontSize', 12);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultAxesXGrid', 'on');
set(groot, 'defaultAxesYGrid', 'on');
set(groot, 'defaultAxesZGrid', 'on');
set(groot, 'defaultAxesXMinorTick', 'on');
set(groot, 'defaultAxesYMinorTick', 'on');
set(groot, 'defaultAxesZMinorTick', 'on');
set(groot, 'defaultAxesXMinorGrid', 'on', 'defaultAxesXMinorGridMode', 'manual');
set(groot, 'defaultAxesYMinorGrid', 'on', 'defaultAxesYMinorGridMode', 'manual');
set(groot, 'defaultAxesZMinorGrid', 'on', 'defaultAxesZMinorGridMode', 'manual');
% To see the modified plot style: get(groot, 'default'), to see all the plot style params: get(groot, 'factory')

display = 0; % 1 to plot additional figures and messages, for debugging, 0 to avoid additional plots.

%% Folder path
path_root = pwd; % Path of this file
path_input = [path_root, '\Inputs\'];
path_output = [path_root, '\Outputs\'];

%% Data of the core, and test (MKS system used)
Date_File = datestr(datetime('Today'), 'yyyy-mm-dd'); % Day when the output files are generated
[Excitation, Material, Shape, N1, N2, Le, Ae, Ve, Date_Test, Info_Test] = ReadCoreInfo([path_input, 'Core_Info.xlsx']);

disp(' '); disp([Excitation, ' ', Material, ' algorithm begins:']); disp(' ');

%% Read the .csv files
[Volt_raw, Curr_raw, Time_raw] = ReadRawData(path_input);
Ndata_raw = length(Volt_raw(:,1)); % Number of datapoints in the whole run
Nsamples = length(Volt_raw(1,:)); % Number of samples per datapoint
disp([num2str(Ndata_raw), ' datapoints with ', num2str(Nsamples), ' samples per datapoint for ', Excitation, ' ', Material, ' read'])

%% Save the raw measurements as a .mat file
DataRaw = struct(...
    'Date_Test', Date_Test,...
    'Date_File', Date_File,...
    'Info_Test', Info_Test,...
    'Material', Material,...
    'Core_Shape', Shape,...
    'Effective_Area', Ae,...
    'Effective_Volume', Ve,...
    'Effective_Length', Le,...
    'Primary_Turns', N1,...
    'Secondary_Turns', N2,...
    'Excitation', Excitation,...
    'Voltage', Volt_raw,...
    'Current', Curr_raw,...
    'Time', Time_raw);

save([path_output, Material, '_', Excitation, '_Raw.mat'], 'DataRaw', '-v7.3');
disp([Material, '_', Excitation, '_Raw.mat file saved'])

%% Eliminating inaccurate data (clipping, voltage, current or power too low, average voltage or current too high)

% Parameters for the detection of inaccurate data
Vmin_absolute = 1; % Minimum peak voltage acceptable
Imin_absolute = 10e-3; % Minimum peak current acceptable
Relative_accuracy = 7.5/100; % DC gain of the scope. 1.5%, then for 1V/div this is +-75mV
Clipping_fraction = 0.1/100; % Deletes the data if more than this fraction is clipped in the scope
Discarded_vector = FindInaccurateData(Volt_raw, Curr_raw, Vmin_absolute, Imin_absolute, Relative_accuracy, Clipping_fraction, display);

% Deleting the inaccurate data
Volt = Volt_raw; Curr = Curr_raw; Time = Time_raw; % Initialization
Volt(Discarded_vector==1,:) = []; % Remove inaccurate data
Curr(Discarded_vector==1,:) = [];
Time(Discarded_vector==1,:) = [];
Ndata = length(Volt(:,1));

%% Obtain the post processed parameters. Constant time step considered only.

% Sampling time (constant for all samples)
Tsampling = (Time(1,Nsamples)-Time(1,1))/(Nsamples-1); % Assuming a constant time step in each sample

% Frequency identification
Freq = IdentificationFrequency(Curr, Tsampling, display); % in Hz, current as the input as it has a lower 3rd harmonic content

% Flux amplitude identification
Flux = IdentificationFlux(Volt/N2/Ae, Freq, Tsampling, display);

% Duty cycle identification for Trapezoidal excitation. d2=d4 only.
Duty_resolution = 0.1; % Duty cycle resolution, this is a critical parameter
Duty_Nthresholds = 3/Duty_resolution; % Rounding and the number of points will determine how accurately the duty detection works (assuming the noise is as large as DeltaV, three times 1/res should be fine)
if Excitation(1)=='S' % Sinusoidal, no duty cycle, --> -1
    DutyP = -ones(Ndata,1); DutyN = -ones(Ndata,1); Duty0 = -ones(Ndata,1);
end
if Excitation(1)=='T' % Trapezoidal
    % Parameters for the estimation of the duty cycle
    [DutyP, DutyN] = IdentificationDutyCycles(Volt, Freq, Tsampling, Duty_resolution, Duty_Nthresholds, display);
    Duty0 = (1-DutyP-DutyN)/2;
end
Duty1 = DutyP;
Duty2 = Duty0;
Duty3 = DutyN;
Duty4 = Duty0;

% Loss density amplitude identification
Loss = IdentificationLoss(Volt, Curr, Freq, Tsampling, display)/Ve; % Volumetric loss in W/m3

disp(['Main parameters extracted for ', Excitation, ' ', Material])

%% Saving the .mat data with the conventional information and the voltage and current, all in SI units

Discard_Algorithm = ['Data discarded: Voltage below ', num2str(Vmin_absolute), ' V; Current below ', num2str(Imin_absolute), ' A; Average values above ', num2str(Relative_accuracy*100), ' percent of the peak value of the sample; Clipped signals for more than ', num2str(Clipping_fraction*100), ' percent of the sample length in a row'];
Freq_Algorithm = ['Frequency estimation based on FFT of the current with a frequency resolution of ', num2str(1/(Nsamples*Tsampling)/1000), ' kHz'];
Flux_Algorithm = 'Flux estimated based on the integral of the voltage minus the switching cycle length moving average and discarding the first and last cycles';
Loss_Algorithm = 'Loss density calculated as the average product of the voltage and the current after removing the first and last switching cycle (removing the average values for the voltage and current) divided by the effective volume';
Duty_Algorithm = ['Duty cycle estimated by dividing the peak to peak voltage in ', num2str(Duty_Nthresholds), ' equally spaced thresholds and checking the percentage of the sample (without the first and last cycle) that is above this threshold with a resolution of', num2str(Duty_resolution)];
Info_Algorithm = [Discard_Algorithm, ' --- ', Freq_Algorithm, ' --- ', Flux_Algorithm, ' --- ', Duty_Algorithm, ' --- ', Loss_Algorithm];

DataMain = struct(...
    'Date_Test', Date_Test,...
    'Date_File', Date_File,...
    'Info_Test', Info_Test,...
    'Info_Algorithm', Info_Algorithm,...
    'Material', Material,...
    'Core_Shape', Shape,...
    'Effective_Area', Ae,...
    'Effective_Volume', Ve,...
    'Effective_Length', Le,...
    'Primary_Turns', N1,...
    'Secondary_Turns', N2,...
    'Excitation', Excitation,...
    'Sampling_Time', Tsampling,...
    'Voltage', Volt,...
    'Current', Curr,...
    'Frequency', Freq,...
    'Flux_Density', Flux,...
    'Duty_1', Duty1,...
    'Duty_2', Duty2,...
    'Duty_3', Duty3,...
    'Duty_4', Duty4,...
    'Power_Loss', Loss); % Volumetric loss

save([path_output, Material, '_', Excitation, '_Main.mat'], 'DataMain', '-v7.3');
disp([Material, '_', Excitation, '_Main.mat file saved'])

%% Ploting the data

% Identification of the run, this is, a shape. Only valid for duty_resolution=0.1, for plot purposes.
if Excitation(1)=='S'
    Run = zeros(Ndata,1);
end
if Excitation(1)=='T' % Trapezoidal
    Run = round(10*DutyP+9*(round(10*Duty0)==1)+16*(round(10*Duty0)==2)+21*(round(10*Duty0)==3)+24*(round(10*Duty0)==4));
end

if Excitation(1)=='S' % Sinusoidal
    figure; % Losses in color in the 3-D plot
    scatter3(Flux*1e3, Freq/1e3, Loss/1e3, 15, log10(Loss/1e3), 'filled');
    c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
    c.Label.String = '$log_{10}(P_V$~[kW/m$^3$]$)$';
    xlabel('AC flux density amplitude [mT]');
    ylabel('Frequency [kHz]');
    zlabel('Loss density [kW/m$^3$]');
    title([Material, ', ', Excitation]);
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
    drawnow();
end
if Excitation(1)=='T' % Trapezoidal
    for n = min(Run):max(Run)
        if sum(Run==n)>1
            figure;
            scatter3(Flux(Run==n)*1e3, Freq(Run==n)/1e3, Loss(Run==n)/1e3, 15, log10(Loss(Run==n)/1e3), 'filled');
            c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
            c.Label.String = '$log_{10}(P_V$~[kW/m$^3$]$)$';
            xlabel('$B_{pk}$ [mT]');
            ylabel('$f$ [kHz]');
            zlabel('$P_V$ [kW/m$^3$]');
            title([Material, ', ', Excitation, ' $d_0$= ', num2str(round(mean(Duty0(Run==n)),1)), '~$d_P$= ', num2str(round(mean(DutyP(Run==n)),1))])
            set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
            drawnow();
        end
    end
end

%% Create the .zip file with the .txt info file and the .cvs reduced voltage and current waveform
fid=fopen([path_output, 'Test_Info.txt'], 'w'); % Clear previous content
fprintf(fid, 'Excitation: '); fprintf(fid, Excitation);
fprintf(fid, ', Sampling time: '); fprintf(fid, num2str(Tsampling*1e9)); fprintf(fid,' ns'); fprintf(fid, '\n');
fprintf(fid, 'Core material: '); fprintf(fid, Material);
fprintf(fid, ', Core shape: '); fprintf(fid, Shape);
fprintf(fid, ', Primary winding: '); fprintf(fid,num2str(N1)); fprintf(fid, ' turns Litz wire');
fprintf(fid, ', Secondary winding: '); fprintf(fid,num2str(N2));  fprintf(fid,' turns Solid wire awg 20'); fprintf(fid, '\n');
fprintf(fid, 'Measurement date (Y/M/D): '); fprintf(fid,Date_Test);  fprintf(fid, '\n');
fprintf(fid, Info_Test);

fclose('all'); % Close this files (and all other files)

Data_fraction = 0.2; % Only saving 20% of the data (this is a cycle at Fmin in this case)

Short_volt = Volt(:,1:ceil(Nsamples*Data_fraction));
Short_curr = Curr(:,1:ceil(Nsamples*Data_fraction));
writematrix(Short_volt, [path_output, 'Raw_Volt_Short.csv']);
writematrix(Short_curr, [path_output, 'Raw_Curr_Short.csv']);

zip([path_output, Material, '_', Excitation, '_Raw_Data_Short'], {...
    [path_output, 'Raw_Volt_Short.csv'],...
    [path_output, 'Raw_Curr_Short.csv'],...
    [path_output, 'Test_Info.txt']});
delete([path_output, 'Raw_Volt_Short.csv']);
delete([path_output, 'Raw_Curr_Short.csv']);
delete([path_output, 'Test_Info.txt']);
disp([Excitation, '_', Material, '_Raw_Data_Short.zip file generated, .csv and .txt files deleted']);

%% Steinmetz parameters for sinusoidal data
if Excitation(1)=='S' % Sinusoidal
    [k_SE, alpha, beta] = GetSteinmetzParameters(Freq, Flux, Loss, display);
    % Calculating the iGSE parameters
    theta_vect = 0:0.0001:2*pi;
    theta_int = cumtrapz(theta_vect, abs(cos(theta_vect)).^alpha.*2^(beta-alpha));
    k_iGSE = k_SE/((2*pi)^(alpha-1)*theta_int(end));
    disp(['iGSE parameters: ki=', num2str(k_iGSE), ', alpha=', num2str(alpha),', beta=', num2str(beta)])
end

%% Outliers weighted by distance, so far only for dstep=0.1 (just to separate the outliers into B and f data)
Closeness_max=0.1; % A tenth of the radious (1 = 1 decade)

Outlier_Factor = GetOutlierFactor(Freq, Flux, Loss, Run, Closeness_max, display);

if display==1
    if  Excitation(1)=='S' % Sinusoidal
        figure;
        scatter3(Flux*1e3, Freq/1e3, Loss/1e3, 15, Outlier_Factor, 'filled');
        c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
        c.Label.String = "Outlier factor [\%]"; caxis([-5 5])
        xlabel('Peak flux density [mT]');
        ylabel('Frequency [kHz]');
        zlabel('Loss density [kW/m$^3$]');
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
    end
    if  Excitation(1)=='T' % Trapzoidal
        for n = min(Run):max(Run)
            if sum(Run==n)>1
                figure
                scatter3(Flux(Run==n)*1e3, Freq(Run==n)/1e3, Loss(Run==n)/1e3, 15, Outlier_Factor(Run==n), 'filled');
                c = colorbar; c.Label.Interpreter = 'latex';c.TickLabelInterpreter='latex';
                c.Label.String="Outlier factor [\%]"; caxis([-5 5])
                xlabel('$B_{pk}$ [mT]');
                ylabel('$f$ [kHz]');
                zlabel('$P_V$ [kW/m$^3$]');
                title([Material, ', ', Excitation, ' $d_0$= ', num2str(round(mean(Duty0(Run==n)),1)), '~$d_P$= ', num2str(round(mean(DutyP(Run==n)),1))])
                set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
                drawnow();
            end    
        end                 
    end
end

%% Saving the light data as .mat with the outlier factor
Info_Outliers = ['Outlier factor: Mismatch between losses and estimated losses based on the local Steinmetz parameters of nearby datapoints in terms of frequency and flux density up to ', num2str(Closeness_max), ' decades far (weighted based on log distance)'];

DataLight = struct(...
    'Date_Test', Date_Test,...
    'Date_File', Date_File,...
    'Info_Test', Info_Test,...
    'Info_Algorithm', Info_Algorithm,...
    'Info_Outliers', Info_Outliers,...
    'Material', Material,...
    'Core_Shape', Shape,...
    'Effective_Area', Ae,...
    'Effective_Volume', Ve,...
    'Effective_Length', Le,...
    'Primary_Turns', N1,...
    'Secondary_Turns', N2,...
    'Excitation_Type', Excitation,...
    'Frequency', Freq,...
    'Flux_Density', Flux,...
    'Duty_1', Duty1,...
    'Duty_2', Duty2,...
    'Duty_3', Duty3,...
    'Duty_4', Duty4,...
    'Power_Loss', Loss,...
    'Outlier_Factor', Outlier_Factor); % In %

save([path_output, Material, '_', Excitation, '_Light.mat'], 'DataLight', '-v7.3')
disp([Material, '_', Excitation, '_Light.mat file saved'])

%% Saving the JSON file for the webpage
DataWebpage = struct(...
    'Material',Material,...
    'Core_Shape',Shape,...
    'Effective_Area',Ae,...
    'Effective_Volume',Ve,...
    'Effective_Length',Le,...
    'Primary_Turns',N1,...
    'Secondary_Turns',N2,...
    'Excitation_Type',Excitation,...
    'Duty_1', single(round(Duty1,1)),...
    'Duty_2', single(round(Duty2,1)),...
    'Duty_3', single(round(Duty3,1)),...
    'Duty_4', single(round(Duty4,1)),...
    'Frequency', single(round(Freq)),...
    'Flux_Density', single(round(Flux,4)),...
    'Power_Loss' ,single(round(Loss,2)),...
    'Outlier_Factor', single(round(Outlier_Factor,2)),...
    'Info_Date', ['Measurement date (Y/M/D): ', Date_Test],...
    'Info_Excitation', ['Excitation: ', Excitation, '; Sampling time: ', num2str(Tsampling*1e9), ' ns'],...
    'Info_Core', ['Core material: ', Material, '; Core shape: ', Shape, '; Primary winding: ', num2str(N1), ' turns Litz wire', '; Secondary winding: ', num2str(N2), ' turns Solid wire awg 20'],...
    'Info_Test', Info_Test);

JSON=jsonencode(DataWebpage);
fprintf(fopen([path_output, Material, '_', Excitation, '_Webpage.json'], 'w'), JSON); fclose('all');
disp([Material, '_', Excitation, '_Webpage.json file saved, with ', num2str(Ndata), ' datapoints'])

%% Saving the data for the scalar-to-scalar NN training, with the outliers removed 
Outlier_Threshold = 5; % Data with an outlier factor above 5 percent is removed
Outliers = abs(Outlier_Factor)>Outlier_Threshold;

DataScalar2Scalar = struct(...
    'Material', Material,...
    'Core_Shape', Shape,...
    'Excitation_Type', Excitation,...
    'Outlier_Threshold', Outlier_Threshold,...
    'Duty_1', single(round(Duty1(Outliers==0),1)),...
    'Duty_2', single(round(Duty2(Outliers==0),1)),...
    'Duty_3', single(round(Duty3(Outliers==0),1)),...
    'Duty_4', single(round(Duty4(Outliers==0),1)),...
    'Frequency',single(round(Freq(Outliers==0))),...
    'Flux_Density', single(round(Flux(Outliers==0),3)),...
    'Power_Loss', single(round(Loss(Outliers==0),2)));

JSON = jsonencode(DataScalar2Scalar);
fprintf(fopen([path_output, Material, '_', Excitation, '_Scalar2Scalar.json'], 'w'), JSON); fclose('all');
disp([Material, '_', Excitation, '_Scalar2Scalar.json file saved, with ', num2str(length(Loss(Outliers==0))), ' datapoints out of ', num2str(Ndata)])

%% Code for the single cycle

% Accurate frequency identification with windowing and zero padding
Freq_Resolution = 10; % In Hz
[Freq_Processed, Error_Freq] = IdentificationFrequencyZeroPaddingAndWindowing(Volt, Freq, Tsampling, Freq_Resolution, display);

% Averaging the Volt and Curr file into a single cycle, only valid for constant timesteps, two cycles per waveform minimum!
Ncycle = 100; % Number of points per cycle for the single-cycle waveform
[Volt_Processed, Curr_Processed, Error_Volt, Error_Curr] = GetSingleCycle(Volt, Curr, Freq_Processed, Tsampling, Ncycle, display);

% Power loss
Loss_Processed=zeros(Ndata,1);
for n=1:Ndata
    Loss_Processed(n)=mean(Volt_Processed(n,:).*Curr_Processed(n,:))/Ve; % Obtain losses again and divide by the effective volume, this time with the averaged waveforms
end
Error_Loss=(Loss-Loss_Processed)./Loss; % Error with respect to the standard method (in both cases without average current)

if display==1
    figure;
    subplot(1,2,1);
    plot(Error_Loss*100, '.k');
    xlabel('Datapoint');
    ylabel('Error in the loss calculation [\%]');
    [~,n_worst] = max(abs(Error_Loss));
    Tcycle = 1/Freq_Processed(n_worst);
    subplot(1,2,2); hold on
    plot((0:floor(Tcycle/Tsampling)-1)*Tsampling*1e6, Volt(n_worst,1:floor(Tcycle/Tsampling)).*Curr(n_worst,1:floor(Tcycle/Tsampling)), 'r')
    plot((0:Ncycle-1)*Tcycle/Ncycle*1e6, Volt_Processed(n_worst,:).*Curr_Processed(n_worst,:), '.b');
    xlabel('Time [us]');
    ylabel('Power [W]');
    legend('First cycle', 'Single cycle');
    sgtitle(['Worst-Case for Losses, Datapoint=', num2str(n_worst)]);
    drawnow();
end

disp(['The average error in losses is ', num2str(round(mean(abs(Error_Loss))*100, 3)), ' % and the peak error is ', num2str(round(max(abs(Error_Loss))*100, 2)), ' %'])

% Obtaining the flux density and field strenght for the averaged waveform
Ndata = length(Volt_Processed(:,1));
Ncycle = length(Volt_Processed(1,:));
B_Processed = zeros(Ndata,Ncycle);
for n=1:Ndata % For each specific datapoint
    Ts_cycle = 1/Freq_Processed(n)/Ncycle; % Sampling time
    Volt_Int = cumtrapz(Volt_Processed(n,:))*Ts_cycle; % Integral of the voltage
    B_Raw = Volt_Int/N2/Ae;
    B_Processed(n,:) = B_Raw-mean(B_Raw); 
end
H_Processed = Curr_Processed*N1/Le;

if display==1
    n_plot=floor(Ndata/2);
    figure;
    subplot(2,3,1);
    plot((0:Ncycle-1)*Ts_cycle*1e6, B_Processed(n_plot,:)*1e3, 'ok');
    xlabel('$t$ [us]');
    ylabel('$B$ [mT]');
    subplot(2,3,4);
    plot((0:Ncycle-1)*Ts_cycle*1e6, H_Processed(n_plot,:), 'ok');
    xlabel('$t$ [us]');
    ylabel('$H$ [A/m]');
    subplot(2,3,[2,3,5,6]);
    plot(H_Processed(n_plot,:), B_Processed(n_plot,:)*1e3, 'ok');
    xlabel('Field Strenght, H [A/m]');
    ylabel('Flux Density, B [mT]');
    sgtitle([Material, ', ', Excitation, ', Datapoint=', num2str(n)]);
    drawnow();
end   

Flux_Processed = (max(B_Processed,[],2)-min(B_Processed,[],2))/2;
Error_Flux = (Flux-Flux_Processed)./Flux; % Error in the amplitude flux with this method

if display==1
    figure;
    subplot(1,2,1);
    plot(Error_Flux*100, '.k');
    xlabel('Datapoint');
    ylabel('Error in the flux amplitude calculation [\%]');
    [~,n_worst] = max(abs(Error_Loss));
    Tcycle=1/Freq_Processed(n_worst);
    Ts_cycle=Tcycle/Ncycle;
    subplot(1,2,2); hold on
    plot((0:floor(Tcycle/Tsampling)-1)*Tsampling*1e6, Volt(n_worst,1:floor(Tcycle/Tsampling)), 'r')
    plot((0:Ncycle-1)*Ts_cycle*1e6, Volt_Processed(n_worst,:), '.b');
    xlabel('Time [us]');
    ylabel('Voltage [V]');
    legend('First cycle', 'Single cycle');
    sgtitle(['Worst-Case for Flux Amplitude, Datapoint=', num2str(n_worst)]);
    drawnow();
end
disp(['The average error in flux amplitude is ', num2str(round(mean(abs(Error_Flux))*100, 3)), ' % and the peak error is ', num2str(round(max(abs(Error_Flux))*100, 2)), ' %'])

%% Eliminating inaccurate data (error with respect to the previous methods)

Error_max = 5/100; % Maximum error for any of the computed errors
Discarded_vector = (Error_Freq>Error_max)+(Error_Flux>Error_max)+(Error_Loss>Error_max)+(Error_Volt>Error_max)+(Error_Curr>Error_max);
Discarded_vector(Discarded_vector~=0) = 1;  % TODO: find a better way to implement this, is there an "or" funcion for vectors?
if display==1
    figure;
    plot(Discarded_vector-1, '.k');
    xlabel('Datapoint');
    ylabel('Inaccurate (0=Yes)');
    axis([1 Ndata -0.5 0.5]);
    drawnow();
end

disp([num2str(sum(Discarded_vector)), ' discarded out of ', num2str(Ndata), ', ', num2str(Ndata-sum(Discarded_vector)), ' remaining'])

% Deleting the inaccurate data
Freq_Cycle = Freq_Processed;
Freq_Cycle(Discarded_vector==1) = []; % Remove inaccurate data
Volt_Cycle = Volt_Processed;
Volt_Cycle(Discarded_vector==1,:) = [];
Curr_Cycle = Curr_Processed;
Curr_Cycle(Discarded_vector==1,:) = [];
Loss_Cycle = Loss_Processed;
Loss_Cycle(Discarded_vector==1) = [];
B_Cycle = B_Processed;
B_Cycle(Discarded_vector==1,:) = [];
H_Cycle= H_Processed;
H_Cycle(Discarded_vector==1,:) = [];
Run_Cycle = Run;
Run_Cycle(Discarded_vector==1,:) = [];
Flux_Cycle = Flux_Processed;
Flux_Cycle(Discarded_vector==1) = [];
DutyP_Cycle = DutyP;
DutyP_Cycle(Discarded_vector==1,:) = [];
DutyN_Cycle = DutyN;
DutyN_Cycle(Discarded_vector==1,:) = [];
Ndata_Cycle = length(Run_Cycle);


%% Outliers weighted by distance for the single cycle

Outlier_Factor_Cycle = GetOutlierFactor(Freq_Cycle, Flux_Cycle, Loss_Cycle, Run_Cycle, Closeness_max, display);

if display==1
    if Excitation(1)=='S' % Sinusoidal
        figure;
        scatter3(Flux_Cycle*1e3, Freq_Cycle/1e3, Loss_Cycle/1e3, 15, Outlier_Factor_Cycle, 'filled');
        c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
        c.Label.String = "Outlier factor [\%]"; caxis([-5 5])
        xlabel('Peak flux density [mT]');
        ylabel('Frequency [kHz]');
        zlabel('Loss density [kW/m$^3$]');
        set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
    end
    if Excitation(1)=='T' % Trapezoidal
        for n = min(Run_Cycle):max(Run_Cycle)
            if sum(Run==n)>1
            figure;
            scatter3(Flux_Cycle(Run_Cycle==n)*1e3, Freq_Cycle(Run_Cycle==n)/1e3, Loss_Cycle(Run_Cycle==n)/1e3, 15, Outlier_Factor_Cycle(Run_Cycle==n), 'filled');
            c = colorbar; c.Label.Interpreter = 'latex';c.TickLabelInterpreter='latex';
            c.Label.String="Outlier factor [\%]"; caxis([-5 5])
            xlabel('$B_{pk}$ [mT]');
            ylabel('$f$ [kHz]');
            zlabel('$P_V$ [kW/m$^3$]');
            title([Material, ', ', Excitation, ' $d_0$= ', num2str(round(mean(DutyN_Cycle(Run_Cycle==n)),1)), '~$d_P$= ', num2str(round(mean(DutyP_Cycle(Run_Cycle==n)),1))])
            set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
            drawnow();
            end
        end
    end
end

%% Saving the reduced data as .mat with the outlier factor

Freq_Algorithm = ['Frequency: Identified based on FFT with zero padding and Hann window with a resolution of ', num2str(Freq_Resolution), ' Hz'];
Cycle_Algorithm = ['Saving a cycle: Averaging all switching cycles but the last one based on the frequency algorithm; using cubic spline interpolation with ', num2str(Ncycle), ' points for the waveforms; the average voltage and current are removed'];
Loss_Algorithm = 'Loss density: calculated as the average product of the voltage and the current divided by the effective volume';
Field_Algorithm = 'B and H field: calculated as the integral of the voltage over the secondary turns and the effective area and as the current times the primary turns over the effective length respectively';
Discard_Algorithm = ['Discarding points: Above ', num2str(Error_max*100) ,' percent mismatch of the difference of the waveform of each cycle with respect to the average, divided by the average value of the absolute signal; Also, discarding points with ', num2str(Error_max*100) ,' percent error in frequency or losses'];
Info_Cycle = [Freq_Algorithm, ' --- ', Cycle_Algorithm, ' --- ', Loss_Algorithm, ' --- ',Field_Algorithm , ' --- ', Discard_Algorithm];

DataCycle = struct(...
    'Date_Test', Date_Test,...
    'Date_File', Date_File,...
    'Info_Test', Info_Test,...
    'Info_Algorithm', Info_Algorithm,...
    'Info_Outliers', Info_Outliers,...
    'Info_Cycle', Info_Cycle,...
    'Material', Material,...
    'Core_Shape', Shape,...
    'Effective_Area', Ae,...
    'Effective_Volume', Ve,...
    'Effective_Length', Le,...
    'Primary_Turns', N1,...
    'Secondary_Turns', N2,...
    'Excitation_Type', Excitation,...
    'Frequency', Freq_Cycle,...
    'Flux_Density', Flux_Cycle,...
    'Duty_P', DutyP_Cycle,...
    'Duty_N', DutyN_Cycle,...
    'Run', Run_Cycle,...
    'Power_Loss', Loss_Cycle,...
    'Voltage', Volt_Cycle,...
    'Current', Curr_Cycle,...
    'B_Field', B_Cycle,...
    'H_Field', H_Cycle,...
    'Outlier_Factor', Outlier_Factor_Cycle);

save([path_output, Material, '_', Excitation, '_Cycle.mat'], 'DataCycle', '-v7.3')
disp([Material, '_', Excitation, '_Cycle.mat file saved'])


%% Create the .zip file with the .txt info file and the .cvs reduced voltage and current waveform
fid=fopen([path_output, 'Cycle_Info.txt'], 'w'); % Clear previous content
fprintf(fid, 'Excitation: '); fprintf(fid, Excitation); fprintf(fid, '\n');
fprintf(fid, 'Core material: '); fprintf(fid, Material);
fprintf(fid, ', Core shape: '); fprintf(fid, Shape);
fprintf(fid, ', Primary turns: '); fprintf(fid,num2str(N1));
fprintf(fid, ', Secondary turns: '); fprintf(fid,num2str(N2)); fprintf(fid, '\n');
fprintf(fid, Info_Algorithm); fprintf(fid, '\n');
fprintf(fid, Info_Cycle); fprintf(fid, '\n');
fprintf(fid, 'Measurement date (Y/M/D): ');fprintf(fid,Date_Test);
fclose('all'); % Close this files (and all other files)

Ts_Cycle = 1./Freq_Cycle/Ncycle;

writematrix(B_Cycle, [path_output, 'B_Field.csv']);
writematrix(H_Cycle, [path_output, 'H_Field.csv']);
writematrix(Ts_Cycle, [path_output, 'Sampling_Time.csv']);

zip([path_output, Material, '_', Excitation, '_Single_Cycle'], {...
    [path_output, 'B_Field.csv'],...
    [path_output, 'H_Field.csv'],...
    [path_output, 'Sampling_Time.csv'],...
    [path_output, 'Cycle_Info.txt']});
delete([path_output, 'B_Field.csv']);
delete([path_output, 'H_Field.csv']);
delete([path_output, 'Sampling_Time.csv']);
delete([path_output, 'Cycle_Info.txt']);
disp([Excitation, '_', Material, '_Single_Cycle.zip file generated, .csv and .txt files deleted']);

%% Generate the .mat file for the sequence2scalar NN combined with randomized initial time
Outliers_Cycle = (abs(Outlier_Factor_Cycle)>Outlier_Threshold);
Flux_Training = B_Cycle(Outliers_Cycle==0,:);
Freq_Training = Freq_Cycle(Outliers_Cycle==0);
Loss_Training = Loss_Cycle(Outliers_Cycle==0);

Ndata_Training = length(Flux_Training(:,1));
Ncycle = length(Flux_Training(1,:));

if display==1
    Flux_Max_Training = max(Flux_Training,[],2);
    Flux_Min_Training = min(Flux_Training,[],2);
    Flux_Amplitude_Training = (Flux_Max_Training-Flux_Min_Training)/2;
    figure; % Losses in color in the 3-D plot
    scatter3(Flux_Amplitude_Training*1e3, Freq_Training/1e3, Loss_Training/1e3, 15, log10(Loss_Training/1e3), 'filled');
    c = colorbar; c.TickLabelInterpreter = 'latex'; c.Label.Interpreter = 'latex';
    c.Label.String = '$log_{10}(P_V$~[kW/m$^3$]$)$';
    xlabel('AC flux density amplitude [mT]');
    ylabel('Frequency [kHz]');
    zlabel('Loss density [kW/m$^3$]');
    title([Material, ', ', Excitation]);
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log'); set(gca, 'ZScale', 'log'); view(2);
    drawnow();
end   

% Randomizing the starting point
Starting_point = floor((101-1e-12)*rand([Ndata_Training,1]));
% 1e-12 to have the same number at 0 and 100 as the other points
% figure; plot(Starting_point, '.k');
Flux_Training_Doubled = [Flux_Training Flux_Training];
Flux_Training_Random = zeros(Ndata_Training,Ncycle);
for n = 1:Ndata_Training
    Flux_Training_Random(n,:) = Flux_Training_Doubled(n,Starting_point(n)+1:Ncycle+Starting_point(n));
end

% Saving the JSON file
DataSequence2Scalar = struct(...
    'Material', Material,...
    'Core_Shape', Shape,...
    'Excitation_Type', Excitation,...
    'Outlier_Threshold', Outlier_Threshold,...
    'Flux',single(round(Flux_Training_Random,4)),...
    'Freq',single(round(Freq_Training)),...
    'Loss',single(round(Loss_Training,1)));

JSON = jsonencode(DataSequence2Scalar);
fprintf(fopen([path_output, Material, '_', Excitation, '_Sequence2Scalar.json'], 'w'), JSON); fclose('all');
disp([Material, '_', Excitation, '_Sequence2Scalar.json file saved, with ', num2str(Ndata_Training), ' datapoints out of ', num2str(length(Outlier_Factor_Cycle))])

%% Datasheet information

% Read the Excel files with the datasheet information
[datasheet_F, datasheet_B, datasheet_T, datasheet_P] = ReadDatasheetData([path_input, 'Datasheet_Data.xlsx']);

if display==1
    figure; hold on;
    scatter3(datasheet_B*1e3, datasheet_F/1e3, datasheet_T, 15, log10(datasheet_P/1e3), 'filled');
    c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
    c.Label.String = '$log_{10}(P_V$~[kW/m$^3$]$)$';
    xlabel('AC flux density amplitude [mT]');
    ylabel('Frequency [kHz]');
    zlabel('Temperature [C]');
    title([Material, ' Datasheet information']);
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    drawnow();
end

% Range of the interpolation
F_min = 50e3;
F_max = 500e3;
F_log_poins = 100;
B_max = 10e-3;
B_min = 300e-3;
B_log_poins = 100;
T_min = 0;
T_max = 125;
T_step = 5;

% Interpolation of the datasheet
linspace_T = (T_min:T_step:T_max)';
logspace_B = 10.^(linspace(log10(B_min),log10(B_max),B_log_poins)');
logspace_F = 10.^(linspace(log10(F_min),log10(F_max),F_log_poins)');

vecspace_F = 0; vecspace_B = 0; vecspace_T = 0; vecspace_P = 0; % Initialization

if datasheet_B>0 % Just checking that the datasheet info is not empty
    interpolated_data = scatteredInterpolant(log10(datasheet_B'), log10(datasheet_F'), datasheet_T', log10(datasheet_P'),'linear','none');

    interpolated_B = repmat(logspace_B, length(logspace_F)*length(linspace_T), 1);

    interpolated_F = repelem(logspace_F, length(logspace_B));
    interpolated_F = repmat(interpolated_F, length(linspace_T), 1);

    interpolated_T = repelem(linspace_T, length(logspace_B)*length(logspace_F));

    interpolated_P = 10.^interpolated_data(log10(interpolated_B), log10(interpolated_F), interpolated_T);
else
    interpolated_B = [0 0];
    interpolated_F = [0 0];
    interpolated_T = [0 0];
    interpolated_P = [0 0];

    disp('The datasheet information is empty')
end

interpolated_B(isnan(interpolated_P)) = [];
interpolated_F(isnan(interpolated_P)) = [];
interpolated_T(isnan(interpolated_P)) = [];
interpolated_P(isnan(interpolated_P)) = [];

if display==1
    figure; hold on;
    scatter3(interpolated_B*1e3, interpolated_F/1e3, interpolated_T, 5, log10(interpolated_P/1e3), 'filled');
    c = colorbar; c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
    c.Label.String = '$log_{10}(P_V$~[kW/m$^3$]$)$';
    xlabel('AC flux density amplitude [mT]');
    ylabel('Frequency [kHz]');
    zlabel('Temperature [C]');
    title([Material, ' Datasheet (interpolated data)']);
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    drawnow();
end

% Saving the datasheet points in a .json file
DataDatasheet=struct(...
    'Material', Material,...
    'Excitation', Excitation,...
    'Frequency', single(round(interpolated_F)),...
    'Flux_Density', single(round(interpolated_B,4)),...
    'Temperature', single(round(interpolated_T)),...
    'Power_Loss', single(round(interpolated_P,4)));

JSON = jsonencode(DataDatasheet);
fprintf(fopen([path_output, Material, '_Datasheet.json'], 'w'), JSON); fclose('all');
disp([Material, '_Datasheet.json file saved, with ', num2str(length(interpolated_P)), ' datapoints'])

%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');
