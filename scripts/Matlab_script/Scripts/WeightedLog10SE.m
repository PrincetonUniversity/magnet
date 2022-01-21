function out = WeightedLog10SE(parameters, in)
%WeightedLog10SE weighted log10 of the losses calulated using Steinmetz equations
%   parameters - Steinmetz k, alpha and beta respectively in a vector
%   in - [Freq Flux Weight] matrix, in Hz and mT respectively (amplitude, not peak to peak)
%   out - log10(Loss)
%
%   It does not treat the points with low losses as less important
%   Pv=k*f^alpha*B^beta; out=weight*log10(Pv)

out = in(:,3).*log10(parameters(1).*in(:,1).^parameters(2).*in(:,2).^parameters(3));

end