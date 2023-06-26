function [Material, Shape, CoreN, N1, N2, Le, Ae, Ve, Date_info, Place_info, Trap_info, Sine_info, Bias_info, Temp_info, Meas_info, Acquisition_info] = ReadTestInfo(path)
%Read the excel file containing the information of the test
%   Material - name of the material of the core
%   Shape - name of the shape of the core
%   CoreN - Number of the core (when more of the same core are to be tested)
%   N1 - number of turns of the primary winding
%   N2 - number of turns of the secondary winding
%   Le - effective length (m)
%   Ae - effective area (m2)
%   Ve - effective volume (m3)
%   Date_info - date of the test (YYYY-MM-DD) 
%   Date_Test - place of the test
%   Trap_info - Information regarding the set-up for piecewise linear
%   excitations
%   Sine_info - Information regarding the set-up for sinusoidal excitations
%   Bias_info - Information regarding the DC bias circuitry
%   Temp_info - Information regarding the set-up to control the temperature
%   Meas_info - Information regarding the DC bias circuitry
%   Acquisition_info - Information regarding the acquisition

% Read the file
if isfile(path)
    info = readcell(path);
    Material = char(info(2,2));
    Shape = char(info(3,2));
    CoreN = cell2mat(info(4,2));
    N1 = cell2mat(info(5,2));
    N2 = cell2mat(info(6,2));
    Le = cell2mat(info(7,2));
    Ae = cell2mat(info(8,2));
    Ve = cell2mat(info(9,2));
    Date_info = char(info(10,2));
    Place_info = char(info(11,2));
    Trap_info = char(info(12,2));
    Sine_info = char(info(13,2));
    Bias_info = char(info(14,2));
    Temp_info = char(info(15,2));
    Meas_info = char(info(16,2));
    Acquisition_info = char(info(17,2));
    disp('.xlsx file containing the test information loaded')
else
    disp('.xlsx file containing the test information not found, quit execution')
    return
end

end