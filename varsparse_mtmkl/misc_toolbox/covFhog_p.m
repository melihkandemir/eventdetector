function [A, B] = covFhog_p(loghyper, x, z)
persistent Dhog_train
if nargin == 0, A = '2'; return; end          % report number of parameters

if isempty(Dhog_train) || length(x) ~= length(Dhog_train)
  load flowers Dhog_train  
  Dhog_train = Dhog_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper(1));                         
sf2 = exp(2*loghyper(2));

if nargin == 2
    A = sf2*exp(-muf*Dhog_train);
elseif nargout == 2                              % compute test set covariances
     if max(z) <= 2040
     load flowers Dhog_train
        A = sf2*exp(-muf*diag(Dhog_train(z,z)));
        B = sf2*exp(-muf*Dhog_train(x,z));
    else
        load flowers Dhog_traintest Dhog_test
        A = sf2*exp(-muf*Dhog_test(z-2040));
        B = sf2*exp(-muf*Dhog_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    if z == 1
        A =  sf2*exp(-muf*Dhog_train).*-(muf*Dhog_train);
    elseif z == 2
        A =  2*sf2*exp(-muf*Dhog_train);
    end
end

