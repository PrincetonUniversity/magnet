function [F, B, T, P] = ReadDatasheetData(path)
%ReadDatasheetData Reads .xlsx files containing the f, B ,T and Pv info
%   path - where csv files are located
%   F - frequency vector in Hz
%   B - AC flux density amplitude vector in T
%   T - temperature vector in Hz
%   P - loss density vector in W/m3
%
%   Data read with GetData Graph Digitizer
%   http://getdata-graph-digitizer.com/ and then stored in an Excel file
%   The data should be written in four columns with any units (f, B ,T and Pv respectively) 

F = []; B = []; T = []; P = []; % Initialization
if isfile(path)
    datasheet = readcell(path);
    F = cell2mat(datasheet(2:end,1))';
    B = cell2mat(datasheet(2:end,2))';
    T = cell2mat(datasheet(2:end,3))';
    P = cell2mat(datasheet(2:end,4))';
else
    disp(['the ',Material,'.xlsx file has not been generated yet'])
end

end