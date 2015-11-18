function model = slrCreate(X, Y, Likelihood, noise) 
%function model = slrCreate(X, Y, Likelihood) 
%  
%   Creates a sparse linear regression model 
%   with spike and slab prior 
%
% Inputs:
%     X input data (each data point  is stored in a row)
%     Y output data (each data pont is stored in a row and Y can have many columns)
%     Likelihood:  only the option 'Gaussian' is supported at the moment 
%     noise (optional): either 'heterosc'  or 'homesc'. Default is is heterosc 
%
% Outputs
%     a structure that contains the model parameters etc. 
%
%  Muchalis K. Titsias, 2011

model.type = 'slr';
model.Likelihood.type =Likelihood;


if nargin < 4 
    noise = 'heterosc';
end

% number of data 
[N D] = size(X); 
model.N = N; 
model.D = D; 
[N Q] = size(Y); 
model.Q = Q; 
model.Y = Y; 
model.X = X;
jitter = 10e-6;
switch model.Likelihood.type
%    
    case 'Gaussian' % standard regression
         %
         model.Likelihood.nParams = Q; % the Q noise variacnes for the outputs  
         model.Likelihood.noise = noise;
         if strcmp(model.Likelihood.noise, 'homosc') 
             model.Likelihood.sigma2 = var(Y(:))/10; 
         else 
             model.Likelihood.sigma2 = var(Y)/10; 
         end
         model.Likelihood.XtX = X'*X; 
         model.Likelihood.diagXtX = diag(model.Likelihood.XtX)'; 
         model.Likelihood.XtXminusDiag = model.Likelihood.XtX - diag(model.Likelihood.diagXtX);
         model.Likelihood.XtY = X'*Y;
         model.Likelihood.YY = sum(Y.*Y,1);
         model.Likelihood.minSigma2 =  jitter*var(Y(:));
         %
%     case 'Probit'  % binary classifcation
%          %
%          % there not adjustable (fixed to one)
%          model.Likelihood.nParams = Q;
%          model.Likelihood.sigma2 = ones(1,Q);  
%          model.Likelihood.XtX = X'*X; 
%          model.Likelihood.diagXtX = diag(model.Likelihood.XtX)'; 
%          model.Likelihood.XtXminusDiag = model.Likelihood.XtX - diag(model.Likelihood.diagXtX);
%         
end     

% Spike and Slab sparse prior 
model.prior.type = 'spikeSlab';
model.prior.typeW = 'normal';
model.prior.muW = 0;
model.prior.sigma2W = 2;
model.prior.typeS = 'binary';
model.prior.alpha = 0.5;

% these prior are currently not used 
% (hyperparameter are found by type II Maximum Likelihood)
model.hyperPrior.alphaType = 'beta';
model.hyperPrior.note = 'hyperpriors are currently not used';
model.hyperPrior.sigma2Type = 'invGammma';
model.hyperPrior.sigma2wType = 'invGammma';

