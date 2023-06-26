function out = Log10SE(parameters, in)
%Log10SE log10 of the losses calulated using Steinmetz equations
%   parameters - Steinmetz k, alpha and beta respectively in a vector
%   in - [Freq Flux] matrix, in Hz and mT respectively (amplitude, not peak to peak)
%   out - log10(Loss)
%
%   It does not treat the points with low losses as less important
%   Pv=k*f^alpha*B^beta; out=log10(Pv)

out = log10(parameters(1).*in(:,1).^parameters(2).*in(:,2).^parameters(3));

end