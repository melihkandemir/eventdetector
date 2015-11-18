% Demo for multiple output regression
clear all;
close all;

addpath misc_toolbox/;
addpath misc_toolbox/gplm/;


train = 1;

% Fix seeds
randn('seed', 1e5);
rand('seed', 1e5);

X = (-10:0.1:10)';
n = length(X);
Xo = X*ones(1,n);
% Kernel matrix
K = exp(-Xo.^2/20).*(4*cos(0.5*(Xo-Xo'))+1*cos(2*(Xo-Xo'))).*exp(-(Xo').^2/20);
%K = (4*cos(0.5*(Xo-Xo'))+1*cos(2*(Xo-Xo')));

% compute Q random functions with this covariance function
Q = 12;
F = chol(K+1e-6*eye(n))'*randn(n,Q);
Y = F + 0.2*randn(n,Q);
Y(:,3) = 0.4*randn(n,1);F(:,3) = 0;
%Y(:,4) = 0.2*randn(n,1);F(:,4) = 0;
Y(:,7) = 0.1*randn(n,1);F(:,7) = 0;

Y2 = Y;

slice = 4;
for q = 1:Q
    Y(find(X==slice*mod(q,20/slice)-10):find(X==slice*mod(q,20/slice)-10+slice),q)=nan;
end

% number of GP latent GP functions 
M = 7;
Iters = 100; 
    
kernels={};
for m=1:M 
    kernels{m} = {'covSEiso_fp'};
end
noise = 'heterosc'; % anything else will mean ouput-specific (heteroscedastic) noise 

% substruct the mean and scale the data 
model = varmkgpCreate(X, Y/sqrt(var(Y(~isnan(Y(:))))), 'Gaussian', kernels, noise);

% keep the substracted mean and scale for prediction purposes
model.meanY = zeros(1,Q);
model.scale = sqrt(var(Y(~isnan(Y(:)))));

%model.Likelihood.sigma2 = 0.01*ones(1,Q); % better initial guess

% training 
% load default options 
load defoptions;
options(1) = 1; % display lower bound during running...
options(2) = 1; % learn kernel hyerparameters (0 for not learning)... 
options(3) = 1; % learn sigma2W hyperprameter (0 for not learning)...
options(4) = 1; % learn likelihood noise parameters sigma2 (0 for not learning)...
options(5) = 1; % learn pi sparse mixing coefficient (0 for not learning)...
options(10) = 1; % use sparsity or not (if not pi is set to 1, is not learned)..
options(11) = Iters; % number of variational EM iterations;

predGP = zeros(n, Q);
varGP = zeros(n, Q);
if train
    %load covlearn_resultsfinal predGP varGp
    for q = 1:Q
        loghyper = minimize([model.GP{1}.logtheta;log(model.scale);0.5*log(model.Likelihood.sigma2(q))], 'gpr', 40, ...
            {'covSum', {'covSEiso','covNoise'}}, X(~isnan(Y(:,q))), Y(~isnan(Y(:,q)),q)/model.scale);
        [mu, S2] = gpr(loghyper,  {'covSum', {'covSEiso','covNoise'}}, X(~isnan(Y(:,q))), Y(~isnan(Y(:,q)),q)/model.scale, X);
        predGP(:,q) = mu*model.scale+model.meanY(q);
        varGP(:,q) = S2*model.scale^2;
    end
    save covlearn_resultsfinal
    [model vardist margLogL] = varmkgpMissDataTrain(model, options);
    save covlearn_resultsfinal
else
    load covlearn_resultsfinal
end

% mean reconstruction of the training data
predF = vardist.muPhi*((vardist.gamma.*vardist.muW)');
predF = predF*model.scale+ones(n,1)*model.meanY;

% Plot observations vs. reconstructions
for q=1:Q
plot(X(isnan(Y(:,q))),Y2(isnan(Y(:,q)),q),'go','MarkerSize',10,'LineWidth',2)
hold on
plot(X,Y(:,q),'k.','MarkerSize',20,'LineWidth',3)
plot(X,predGP(:,q),'b','LineWidth',2);
plot(X,predF(:,q),'r--','LineWidth',4)
hold off
set(gca,'FontName','Times New Roman','FontSize',10)
print('-depsc',  ['toyexample' num2str(q)])
end

for q=1:Q
subplot(4,3,q); 
plot(X(isnan(Y(:,q))),Y2(isnan(Y(:,q)),q),'bx',X,Y(:,q),'b.')
hold on
plot(X,predGP(:,q),'g','LineWidth',2);
plot(X,predF(:,q),'r','LineWidth',2)
hold off
end
print -depsc  toyexample

% Covariance matrix reconstruction
%figure
%[U,S]=svd(K);U=U*sqrt(S);
%R=vardist.muPhi\U(:,1:M);
%subplot(1,2,1); image(K*50);
%subplot(1,2,2); image(vardist.muPhi*R*R'*vardist.muPhi'*50);

% MSE of independent GPs and multitask learning

MSE = zeros(2,Q);
for q = 1:Q
    MSE(:,q) = [sum( (Y2(isnan(Y(:,q)),q) - predGP(isnan(Y(:,q)),q)).^2);
                sum( (Y2(isnan(Y(:,q)),q) - predF (isnan(Y(:,q)),q)).^2)];
end
MSE
% How much does each latent function contribute to the overall mean?
err = zeros(M,1);
for m = 1:M
    err(m) = norm(model.scale*vardist.muPhi(:,m)*((vardist.gamma(:,m).*vardist.muW(:,m))'),'fro');
end
effM = find(sort(err,'descend').^2./sum(err.^2)<0.001,1)-1
err
%figure
%plot(sort(err,'descend'))
