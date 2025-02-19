function [Volt, Curr, Time, Params] = ReadRawData(path)
%ReadRawData Reads the csv files from the specified folder.
%   path - file location
%   Volt - voltage matrix, single precision
%   Curr - current matrix, single precision
%   Time - time matrix, single precision
%   Param - Parameter matrix (Hdc, d0, dP, freq, flux)
%   csv files should be named 'X_X_X_Data_Volt.csv',
%   All files have to contain the same number of samples per point.
%   Each row of the output is a measurement

% Initialization, data length unknown beforehand
Volt = []; Curr = []; Time = []; Params = [];
flag_files=0; % Just to warn if there are no files to be read


for file_type = 0:1
    file_ref = [num2str(file_type), '_1_1_'];
    for file_bias = 1:10
        for file_run = 1:25
            file_ref = [num2str(file_type), '_', num2str(file_bias), '_', num2str(file_run), '_'];

            file_volt = [path, file_ref, 'Data_Volt.csv'];
            file_curr = [path, file_ref, 'Data_Curr.csv'];
            file_time = [path, file_ref, 'Data_Time.csv'];
            file_params = [path, file_ref, 'Parameters.csv'];

            if isfile([path, file_ref, 'Data_Volt.csv'])
                Volt = vertcat(Volt, single(load(file_volt))); % Single precision, as it does not affect quality of the data
                Curr = vertcat(Curr, single(load(file_curr))); % All files are merged into a single variable
                Time = vertcat(Time, single(load(file_time))); % All files must have the same number of samples per point
                Params = vertcat(Params, load(file_params));
                disp(['raw ', num2str(file_type), '-', num2str(file_bias), '-', num2str(file_run), ' files loaded']) 
                flag_files = 1;
            end
        end
        file_ref = [num2str(file_type), '_', num2str(file_bias), '_1_'];
    end
    file_ref = [num2str(file_type), '_1_1_'];
end

if flag_files==0 
    disp('No .csv files found')
    %return
else    
    disp('All .csv files loaded')
end

end