% DEMO_LINREG: SPARSE  LINEAR REGRESSION  (with this you can do your own demo)

% Load the data
X = randn(100,10);
Y = randn(100,5);

% create the model 
model = slrCreate(X, Y,  'Gaussian');
   
% run paired variational mean field inference
iters = 300;  % number of EM iterations
dispF = 1;    % display lower bound during optimization
[model vardist, F] = slrPairMF(model, iters, dispF);
