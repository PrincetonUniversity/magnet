function out = FFTshort(in, harmonic_max)
%FFTshort computes the fft of the function for an even number of samples, and
%outputs only the first samples of interest
%   in - either the voltage or the current matrix (single precision)
%   n_max - lenght of the output sample (10 -> the first 10 harmonics)
%   out - amplitude of each harmonic, DC amplitude removed
%
%   The first frequency is the inverse of the sample total time.

fft_sample = abs(fft(in)/length(in)); % Module of the fft
% The firt value of the fft is the DC, then it goes to the max frequency up
% to half the length, then from the max frequency to DC again (but not DC)
fft_positive = 2*fft_sample(2:length(in)/2+1); % Spectrum by two, but for the DC point
out = fft_positive(1:harmonic_max);
end
