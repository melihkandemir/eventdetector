function [A, B] = covFhsv_p(loghyper, x, z)
persistent Dhsv_train
if nargin == 0, A = '2'; return; end          % report number of parameters

if isempty(Dhsv_train) || length(x) ~= length(Dhsv_train)
  load flowers Dhsv_train  
  Dhsv_train = Dhsv_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper(1));                         
sf2 = exp(2*loghyper(2));

if nargin == 2
    A = sf2*exp(-muf*Dhsv_train);
elseif nargout == 2                              % compute test set covariances
     if max(z) <= 2040
     load flowers Dhsv_train
        A = sf2*exp(-muf*diag(Dhsv_train(z,z)));
        B = sf2*exp(-muf*Dhsv_train(x,z));
    else
        load flowers Dhsv_traintest Dhsv_test
        A = sf2*exp(-muf*Dhsv_test(z-2040));
        B = sf2*exp(-muf*Dhsv_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    if z == 1
        A =  sf2*exp(-muf*Dhsv_train).*-(muf*Dhsv_train);
    elseif z == 2
        A =  2*sf2*exp(-muf*Dhsv_train);
    end
end

