function [A, B] = covFsiftbdy_p(loghyper, x, z)
persistent Dsiftbdy_train
if nargin == 0, A = '2'; return; end          % report number of parameters

if isempty(Dsiftbdy_train) || length(x) ~= length(Dsiftbdy_train)
  load flowers Dsiftbdy_train  
  Dsiftbdy_train = Dsiftbdy_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper(1));                         
sf2 = exp(2*loghyper(2));

if nargin == 2
    A = sf2*exp(-muf*Dsiftbdy_train);
elseif nargout == 2                              % compute test set covariances
     if max(z) <= 2040
     load flowers Dsiftbdy_train
        A = sf2*exp(-muf*diag(Dsiftbdy_train(z,z)));
        B = sf2*exp(-muf*Dsiftbdy_train(x,z));
    else
        load flowers Dsiftbdy_traintest Dsiftbdy_test
        A = sf2*exp(-muf*Dsiftbdy_test(z-2040));
        B = sf2*exp(-muf*Dsiftbdy_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    if z == 1
        A =  sf2*exp(-muf*Dsiftbdy_train).*-(muf*Dsiftbdy_train);
    elseif z == 2
        A =  2*sf2*exp(-muf*Dsiftbdy_train);
    end
end

