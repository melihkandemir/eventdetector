function [A, B] = covFsiftint_p(loghyper, x, z)
persistent Dsiftint_train
if nargin == 0, A = '2'; return; end          % report number of parameters

if isempty(Dsiftint_train) || length(x) ~= length(Dsiftint_train)
  load flowers Dsiftint_train  
  Dsiftint_train = Dsiftint_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper(1));                         
sf2 = exp(2*loghyper(2));

if nargin == 2
    A = sf2*exp(-muf*Dsiftint_train);
elseif nargout == 2                              % compute test set covariances
     if max(z) <= 2040
     load flowers Dsiftint_train
        A = sf2*exp(-muf*diag(Dsiftint_train(z,z)));
        B = sf2*exp(-muf*Dsiftint_train(x,z));
    else
        load flowers Dsiftint_traintest Dsiftint_test
        A = sf2*exp(-muf*Dsiftint_test(z-2040));
        B = sf2*exp(-muf*Dsiftint_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    if z == 1
        A =  sf2*exp(-muf*Dsiftint_train).*-(muf*Dsiftint_train);
    elseif z == 2
        A =  2*sf2*exp(-muf*Dsiftint_train);
    end
end

