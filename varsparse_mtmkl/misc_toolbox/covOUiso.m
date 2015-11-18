function [A, B] = covOUiso(loghyper, x, z)

% Matern covariance function with nu = 3/2 and isotropic distance measure. The
% covariance function is:
%
% k(x^p,x^q) = s2f * (1 + sqrt(3)*d(x^p,x^q)) * exp(-sqrt(3)*d(x^p,x^q))
%
% where d(x^p,x^q) is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), P is ell
% times the unit matrix and sf2 is the signal variance. The hyperparameters
% are:
%
% loghyper = [ log(ell)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)

if nargin == 0, A = '1'; return; end

[n, D] = size(x);
ell = exp(loghyper(1));
sf2 = 1;

x = sqrt(1)*x/ell;

if nargin == 2                                      % compute covariance matrix
  A = sqrt(sq_dist(x'));
  K = sf2*exp(-A);
  A = K;
elseif nargout == 2                              % compute test set covariances
  z = sqrt(1)*z/ell;
  A = sf2;
  B = sqrt(sq_dist(x',z'));
  B = sf2*exp(-B);
else                                              % compute derivative matrices
  if z == 1
    A = sf2*sqrt(sq_dist(x')).*exp(-sqrt(sq_dist(x')));
  end
end
