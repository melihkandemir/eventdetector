function [A, B] = covFsiftint(loghyper, x, z)
persistent Dsiftint_train
if nargin == 0, A = '1'; return; end          % report number of parameters

if isempty(Dsiftint_train) || length(x) ~= length(Dsiftint_train)
  load flowers Dsiftint_train
  Dsiftint_train = Dsiftint_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper);                         % K^muf

if nargin == 2
    A = exp(-muf*Dsiftint_train);
elseif nargout == 2                              % compute test set covariances
    if max(z) <= 2040
        load /local/mtitsias/michalis/flowers Dsiftint_train
        A = exp(-muf*diag(Dsiftint_train(z,z)));
        B = exp(-muf*Dsiftint_train(x,z));
    else
        load /local/mtitsias/michalis/flowers Dsiftint_traintest Dsiftint_test
        A = exp(-muf*Dsiftint_test(z-2040));
        B = exp(-muf*Dsiftint_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    A =  exp(-muf*Dsiftint_train).*-(muf*Dsiftint_train);
end

