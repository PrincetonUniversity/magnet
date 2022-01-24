function [Volt, Curr, Time] = ReadRawData(path)
%ReadRawData Reads the csv files from the specified folder.
%   path - file location
%   Volt - voltage matrix, single precision
%   Curr - current matrix, single precision
%   Time - time matrix, single precision
%
%   csv files should be named 'Data_Volt_X.csv',
%   All files have to contain the same number of samples per point.
%   Each row of the output is a measurement

% Initialization, data length unknown beforehand
Volt = []; Curr = []; Time = [];
flag_files=0; % Just to warn if there are no files to be read
for file_idn = 0:100 % Max number of files, edit if required
    file_volt = [path, 'Data_Volt_', num2str(file_idn) ,'.csv'];
    file_curr = [path, 'Data_Curr_', num2str(file_idn) ,'.csv'];
    file_time = [path, 'Data_Time_', num2str(file_idn) ,'.csv'];
    if isfile(file_volt) && isfile(file_curr) && isfile(file_time) % Check if the files exist
        Volt = vertcat(Volt, single(load(file_volt)));
        disp(['volt file ', num2str(file_idn), ' loaded']) % Single precision, as it does not affect quality of the data
        Curr = vertcat(Curr, single(load(file_curr)));
        disp(['curr file ', num2str(file_idn), ' loaded']) % All files are merged into a single variable
        Time = vertcat(Time, single(load(file_time)));
        disp(['time file ', num2str(file_idn), ' loaded']) % All files must have the same number of samples per point
        flag_files = 1;
    else
        continue % File does not exist.
    end
end
if flag_files==0 
    disp('No input .csv files found, quit execution')
    return
else    
    disp('All .csv files loaded')
end
end