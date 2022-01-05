function [Waveform, Material, Shape, N1, N2, Le, Ae, Ve, Date_Test, Setup] = ReadCoreInfo(path)
%IdentificationLoss Find the power losses
%   Waveform - type of excitation, either Sinusoidal or Trapezoidal
%   Material - name of the material of the core
%   Shape - name of the shape of the core
%   N1 - number of turns of the primary winding
%   N2 - number of turns of the secondary winding
%   Le - effective length (m)
%   Ae - effective area (m2)
%   Ve - effective volume (m3)
%   Date_Test - date of the text (YYYY-MM-DD) 
%   Setup - Information regarding the test to be included in the files 

% Read the file
if isfile(path)
    info = readcell(path);
    Waveform = char(info(2,2));
    Material = char(info(3,2));
    Shape = char(info(4,2));
    N1 = cell2mat(info(5,2));
    N2 = cell2mat(info(6,2));
    Le = cell2mat(info(7,2));
    Ae = cell2mat(info(8,2));
    Ve = cell2mat(info(9,2));
    Date_Test = char(info(10,2));
    Setup = char(info(11,2));
    
else
    disp('the .xlsx file has not been found')
end

end