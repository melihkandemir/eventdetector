% Demo for Flowers 102-class problem
clear all;
addpath misc_toolbox/;
addpath misc_toolbox/gplm/;

randn('seed', 2e5);
rand('seed', 2e5);

load flowers X_tr T_tr X_tst T_tst

NumberofClasses = 102;
% number of GP basis functions
M = 52;

% Select subset of data points to be used
subsettr = ismember(T_tr, (1:NumberofClasses)');
subsettst = ismember(T_tst, (1:NumberofClasses)'); 
X_tr = X_tr(subsettr,:);
T_tr = T_tr(subsettr);
X_tst = X_tst(subsettst,:);
T_tst = T_tst(subsettst);

% Expand class labels to binary vectors
n =size(T_tr,1);
Q = max(T_tr);
T_train = -ones(n,Q);
for k =1:n
    T_train(k,T_tr(k)) = +1;
end
T_tr = T_train;
T_test = -ones(size(T_tst,1),Q);
for k =1:size(T_tst,1);
    T_test(k,T_tst(k)) = +1;
end
T_tst = T_test;

kernels =[];
muW = zeros(Q,M);
for k = 1:4:M
    kernels{k} = {'covFhog'};
    kernels{k+1} = {'covFhsv'};
    kernels{k+2} = {'covFsiftbdy'};
    kernels{k+3} = {'covFsiftint'};
end

noise = 'heterosc';
model = varmkgpCreate(X_tr, T_tr, 'Gaussian', kernels, noise);

% load default options 
load defoptions;
options(1) = 1; % display lower bound during running...
options(2) = 1; % learn kernel hyerparameters (0 for not learning)... 
options(3) = 1; % learn sigma2W hyperprameter (0 for not learning)...
options(4) = 0; % learn likelihood noise parameters sigma2 (0 for not learning)...
options(5) = 1; % learn pi sparse mixing coefficient (0 for not learning)...
options(10) = 1; % use sparsity or not (if not pi is set to 1, is not learned)..
options(11) = 10; % number of variational EM iterations;

% do few steps with sifex sigma2 for initialization 
model.Likelihood.sigma2 = 0.01;
[model vardist margLogL] = varmkgpTrain(model, options);

% continue with learning sigma2
options(4) = 1; % learn likelihood noise parameters sigma2 (0 for not learning)...
options(11) = 50; % number of variational EM iterations;
[model vardist margLogL] = varmkgpTrain(model, options, vardist);


% Test predictions
outTs = varmkgpPredict(model, vardist, X_tst);
save flowers_resultsmini
disp('Test error in Sparse trained Probit model');
if  strcmp(model.Likelihood.type, 'Gaussian')
    outTs = exp(outTs)./repmat(sum(exp(outTs),2),1,Q);
end
auc=zeros(1,model.Q);for q=1:model.Q, [tpr,fpr] = roc(T_tst(:,q)'>0, outTs(:,q)'); auc(q)=trapz(fpr, tpr);end
mean(auc)

% Train error
disp('Train error in Sparse trained Probit model');
outTr = varmkgpPredict(model, vardist, model.X);
if strcmp(model.Likelihood.type, 'Gaussian')  
   outTr = exp(outTr)./repmat(sum(exp(outTr),2),1,Q);
end
aucTr=zeros(1,model.Q);for q=1:model.Q, [tprTr,fprTr] = roc(T_tr(:,q)'>0, outTr(:,q)'); aucTr(q)=trapz(fprTr, tprTr);end
mean(aucTr)

% compute confusion matrix 
Conf = zeros(NumberofClasses,NumberofClasses);
for i=1:size(T_tst,1)
  [maxP C] = max(outTs(i,:));
  [maxY CC] = max(T_tst(i,:));
  Conf(CC, C) = Conf(CC, C) + 1; 
end

disp('Total multi-class accuracy')
sum(diag(Conf))/size(T_tst,1)

save flowers_result model vardist margLogL auc aucTr Conf;

