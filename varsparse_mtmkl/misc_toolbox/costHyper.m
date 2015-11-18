function [out1, out2] = costHyper(logtheta, covfunc, phiphit, x)

[n, D] = size(x);

if eval(feval(covfunc{:})) ~= size(logtheta, 1)
  error('Error: Number of parameters do not agree with covariance function')
end

K = feval(covfunc{:}, logtheta, x);     % compute  covariance matrix

L = chol(K)';                           % cholesky factorization of the covariance
invK = L'\(L\eye(n));                   % Inverse

out1 = 0.5*sum(sum(invK.*phiphit)) + sum(log(diag(L)));

if nargout == 2                      % ... and if requested, its partial derivatives
    out2 = zeros(size(logtheta));    % set the size of the derivative vector
    W = invK-invK*phiphit*invK;                % precompute 
    for i = 1:length(out2)
        out2(i) = sum(sum(W.*feval(covfunc{:}, logtheta, x, i)))/2;
    end
end