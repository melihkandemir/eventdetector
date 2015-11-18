function [A, B] = covFhog(loghyper, x, z)
persistent Dhog_train
if nargin == 0, A = '1'; return; end          % report number of parameters

if isempty(Dhog_train) || length(x) ~= length(Dhog_train)
  load flowers Dhog_train
  Dhog_train = Dhog_train(x,x);
end

[n D] = size(x);
muf = exp(loghyper);                         % K^muf

if nargin == 2
    A = exp(-muf*Dhog_train);
elseif nargout == 2                              % compute test set covariances
    if max(z) <= 2040
        load /local/mtitsias/michalis/flowers Dhog_train
        A = exp(-muf*diag(Dhog_train(z,z)));
        B = exp(-muf*Dhog_train(x,z));
    else
        load /local/mtitsias/michalis/flowers Dhog_traintest Dhog_test
        A = exp(-muf*Dhog_test(z-2040));
        B = exp(-muf*Dhog_traintest(x,z-2040));
    end
else                                                % compute derivative matrix
    A =  exp(-muf*Dhog_train).*-(muf*Dhog_train);
end

