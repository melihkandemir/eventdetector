function [A, B] = covFhsv(loghyper, x, z)
persistent Dhsv_train
if nargin == 0, A = '1'; return; end          % report number of parameters

if isempty(Dhsv_train) || length(x) ~= length(Dhsv_train)
  load flowers Dhsv_train
  Dhsv_train = Dhsv_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper);                         % K^muf

if nargin == 2
    A = exp(-muf*Dhsv_train);
elseif nargout == 2                              % compute test set covariances
    if max(z) <= 2040
        load /local/mtitsias/michalis/flowers Dhsv_train
        A = exp(-muf*diag(Dhsv_train(z,z)));
        B = exp(-muf*Dhsv_train(x,z));
    else
        load /local/mtitsias/michalis/flowers Dhsv_traintest Dhsv_test
        A = exp(-muf*Dhsv_test(z-2040));
        B = exp(-muf*Dhsv_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    A =  exp(-muf*Dhsv_train).*-(muf*Dhsv_train);
end

