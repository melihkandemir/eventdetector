function model = varmkgpCreate(X, Y, Likelihood, kernels, noise, indexPresent)
%function model = varmkgpCreate(X, Y, Likelihood, kernels, noise, indexPresent)
%
%  Michalis K. Titsias, 2011

model.type = 'mkgpmodel';
model.Likelihood.type = Likelihood;

% number of data 
[n D] = size(X); 
[n Q] = size(Y); 
model.n = n; 
model.D = D;

% number of GP latent functions (as many as the kernels provided)
model.M = size(kernels,2); 

model.Q = Q; 
model.Y = Y; 
model.X = X;

if nargin < 6
    indexPresent = [];
end

% if Y has not been stored as sparse
if  isempty(indexPresent)
   % check if there exist missing values
   indexPresent = ones(size(Y));
   %sigma2q = zeros(1, model.Q); 
   YY = zeros(1, model.Q); 
   for q= 1:model.Q
       indexMissing = isnan(model.Y(:, q));
       indexPresent(indexMissing,q) = 0;
       %sigma2q(q) = var( model.Y(indexPresent(:,q)==1, q) );
       YY(q) = sum(model.Y(indexPresent(:,q)==1, q).*Y(indexPresent(:,q)==1, q), 1);
   end
   model.indexPresent = sparse(indexPresent); 
   model.numPresent = sum(indexPresent, 1);
   % make all NaN zeros
   model.Y(~model.indexPresent)=0;
else
   % Y is given as sparse matrix 
   model.indexPresent = indexPresent; 
   model.numPresent = sum(model.indexPresent, 1);
   %sigma2q = zeros(1, model.Q); 
   YY = zeros(1, model.Q); 
   for q= 1:model.Q
       %sigma2q(q) = var( model.Y(model.indexPresent(:,q)==1, q) );
       YY(q) = sum(model.Y(model.indexPresent(:,q)==1, q).*Y(model.indexPresent(:,q)==1, q), 1);
   end
end


jitter = 10e-6; 
switch model.Likelihood.type
%    
    case 'Gaussian' % standard regression
         %
         if nargin == 4
             noise = 'heterosc';
         end
         model.Likelihood.noise = noise; 
         model.Likelihood.nParams = Q; % the Q noise variacnes for the outputs  
         if strcmp(model.Likelihood.noise, 'homosc') 
            model.Likelihood.sigma2 = double(var(Y(model.indexPresent==1)))/10; 
         else 
            model.Likelihood.sigma2 = (double(var(Y(model.indexPresent==1)))/10)*ones(1,model.Q); 
            model.Likelihood.sigma2(model.Likelihood.sigma2<0.01) = 0.01; 
         end
         model.Likelihood.minSigma2 =  jitter;
         model.Likelihood.nParams = size(model.Likelihood.sigma2, 2);
         % create a 
         model.Likelihood.YY = YY;
         %
    case 'Probit'  % binary classifcation
         %
         % there not adjustable (fixed to one)
         model.Likelihood.nParams = Q;
         model.Likelihood.sigma2 = ones(1,Q);  
%         
end     


model.existBias =0;
% for each latent function create the GP 
for j=1:model.M
%   
    % in Rasmussen format
    model.GP{j}.covfunc = kernels{j}; 
    if ischar(model.GP{j}.covfunc), model.GP{j}.covfunc = cellstr(model.GP{j}.covfunc); end % convert to cell if needed
    % hyperprameters initialization/definition in Rasmussen format 
    % for all poissble kernels in the toolbox
    switch char(model.GP{j}.covfunc{1})
        case 'covSEard'       
        %  
            dd = log((max(X)-min(X))'/2);
            dd(dd==-Inf)=0;
            model.GP{j}.logtheta(1:D,1) = dd;
               model.GP{j}.logtheta(D+1,1) = 0;
               model.GP{j}.logtheta(D+1,1) = 0;
               model.GP{j}.nParams = length(model.GP{j}.logtheta);
               model.GP{j}.constDiag = 1; 
        case 'covSEiso'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta(1,1) = mean(dd);
           model.GP{j}.logtheta(2,1) = 0;
           model.GP{j}.nParams = length(model.GP{j}.logtheta);
        case 'covSEiso_fp'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);
        case 'covSEiso_fp_view1'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);
        case 'covSEiso_fp_view2'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);      
        case 'covSEiso_fp_view3'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);    
        case 'covSEiso_fp_view4'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);    
        case 'covSEiso_fp_view5'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);    

        case 'covSEiso_fp_view6'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);               
        case 'covMatern3iso'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta(1,1) = mean(dd);
           model.GP{j}.logtheta(2,1) = 0;
           model.GP{j}.nParams = length(model.GP{j}.logtheta);
        %   
        case 'covOUiso'
        %     
           dd = log((max(X)-min(X))'/2);
           dd(dd==-Inf)=0; 
           model.GP{j}.logtheta(1,1) = mean(dd);
           model.GP{j}.nParams = length(model.GP{j}.logtheta);
        case 'covImage'
           model.GP{j}.logtheta = [];
           model.GP{j}.nParams = length(model.GP{j}.logtheta);   
        case 'covFhog'
            load flowers Dhog_train
            model.GP{j}.logtheta(1) =  -log(mean(mean(Dhog_train)));
            clear Dhog_train
            %model.GP{j}.logtheta(1) =  log(0.09);
            model.GP{j}.nParams = length(model.GP{j}.logtheta);
            model.GP{j}.constDiag = 1; 
        case 'covFhsv'
            load flowers Dhsv_train
            model.GP{j}.logtheta(1) =  -log(mean(mean(Dhsv_train)));
            clear Dhsv_train
            %model.GP{j}.logtheta(1) =  log(0.04);
            model.GP{j}.nParams = length(model.GP{j}.logtheta);
            model.GP{j}.constDiag = 1; 
        case 'covFsiftbdy'
            load flowers Dsiftbdy_train
            model.GP{j}.logtheta(1) =  -log(mean(mean(Dsiftbdy_train)));
            clear Dsiftbdy_train
            %model.GP{j}.logtheta(1) = log(0.065);
            model.GP{j}.nParams = length(model.GP{j}.logtheta);
            model.GP{j}.constDiag = 1; 
        case 'covFsiftint'
            load flowers Dsiftint_train
            model.GP{j}.logtheta(1) =  -log(mean(mean(Dsiftint_train)));
            clear Dsiftint_train
            %model.GP{j}.logtheta(1) =  log(0.035);
            model.GP{j}.nParams = length(model.GP{j}.logtheta);
            model.GP{j}.constDiag = 1; 
        case 'const'
           % constant function with value one 
           model.GP{j}.logtheta = 0;
           model.GP{j}.nParams = 0;
           model.existBias =1;
        % the rest kernel go here   
        case 'stadNormal'
           % standard normal Gaussian prior 
           model.GP{j}.logtheta = 0;
           model.GP{j}.nParams = 0; 
        otherwise
           char(model.GP{j}.covfunc{1})
           error('Unknown covariance type')
    end
end 


% Spike and Slab sparse prior 
model.prior.type = 'spikeSlab';
model.prior.typeW = 'normal';
model.prior.muW = 0;
model.prior.sigma2W = 1;%*ones(1,Q);
model.prior.typeS = 'binary';
model.prior.pi = 0.5;
