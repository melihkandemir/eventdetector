function [A, B] = covFsiftbdy(loghyper, x, z)
persistent Dsiftbdy_train
if nargin == 0, A = '1'; return; end          % report number of parameters

if isempty(Dsiftbdy_train) || length(x) ~= length(Dsiftbdy_train)
  load flowers Dsiftbdy_train
  Dsiftbdy_train = Dsiftbdy_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper);                         % K^muf

if nargin == 2
    A = exp(-muf*Dsiftbdy_train);
elseif nargout == 2                              % compute test set covariances
    if max(z) <= 2040
        load /local/mtitsias/michalis/flowers Dsiftbdy_train
        A = exp(-muf*diag(Dsiftbdy_train(z,z)));
        B = exp(-muf*Dsiftbdy_train(x,z));
    else
        load /local/mtitsias/michalis/flowers Dsiftbdy_traintest Dsiftbdy_test
        A = exp(-muf*Dsiftbdy_test(z-2040));
        B = exp(-muf*Dsiftbdy_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    A =  exp(-muf*Dsiftbdy_train).*-(muf*Dsiftbdy_train);
end

