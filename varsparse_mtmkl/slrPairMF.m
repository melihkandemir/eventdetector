function [model vardist FF] = slrPairMF(model,iters, dispF, vardist)
%function [model vardist FF] = slrPairMF(model,T, dispF, vardist)
%
% Inputs: 
%    model: model sctyre for sparse linear model created from slrCreate
%    iters: maximum number of variational EM iterations 
%    dispF: display value of the lower bound during optimization
%              if 1 will dispaly, otheriwise no)
%    vardist (optional): stucture that store the variational parameters 
%                It has two fields: 'gamma' => spike-and-slab variationsl posteriors
%                                          'muW'  => variational means for
%                                          the weights
%
% Ouputs: 
%     model: updated mdoel with optimized hyperparameters
%     vardist: optimized variational distrbution
%     FF: full path of value of the variational lower bound (only if dispF=1, otherwise FF=[])
%
% Michalis K. Titsias, 2011


Q = model.Q; 
D = model.D; 
N = model.N;

debug = 0; 
dispEvery = 10;

% if variational distrbutino is not given as input, then initilize it here
if nargin <=3
    % INITIALIZE FACTOR  q(s_i=1) 
    % (around 0.5)
    gamma = 0.5*ones(Q, D) + 0.001*randn(Q, D); 
    gamma(gamma<=0.2)=0.2;
    gamma(gamma>=0.8)=0.8;
    vardist.gamma = gamma;
    % INITIALIZE FACTOR q(w_m | s_m =1) = N(w_m| muW_m,  sigma2_wm)
    vardist.muW = 0.1*randn(Q, D);              % initial variational mean
end


diagXtX = model.Likelihood.diagXtX;
XtXminusDiag = model.Likelihood.XtXminusDiag;

Fold = -Inf;
% start variational EM
for it=1:iters
%    
    logAlpha = log(model.prior.alpha./(1-model.prior.alpha));
    sigma2Sigmaw = model.Likelihood.sigma2./model.prior.sigma2W;
    logSigma2Sigma2w = 0.5*log(sigma2Sigmaw);

    % E-STEP --- START
    if strcmp(model.Likelihood.noise, 'homosc')
       xxsigma = repmat(diagXtX, Q, 1) + sigma2Sigmaw;     
       vardist.sigma2W = model.Likelihood.sigma2./xxsigma;
    else
       xxsigma = repmat(diagXtX, Q, 1) + repmat(sigma2Sigmaw', 1, D);     
       vardist.sigma2W = repmat( model.Likelihood.sigma2', 1, D)./xxsigma;
    end
    %
    for m=1:D
    %   
        setMinus = [1:m-1,  (m+1):D]; 
        
        b = (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))*model.Likelihood.XtX(m,setMinus)';
        diff = model.Likelihood.XtY(m, :) - b';     
        um = logAlpha + logSigma2Sigma2w - 0.5*log(xxsigma(:,m)') ...
             +  (0.5./model.Likelihood.sigma2).*((diff.^2)./(xxsigma(:,m)'));
       
        % Update variational distribution
        vardist.gamma(:,m) = 1./(1+exp(-um)); 
        vardist.muW(:,m) = diff'./xxsigma(:,m); 
        
        if debug==1
             F= slrLowerBound(model, vardist);
             if (F-Fold) < 0 
               fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(w_m,s_m))\n',it, F, Fold,  F-Fold);
             end
             Fold = F;
        end
    %     
    end
    % E-STEP --- END
    
    
    % M-STEP --- START
    % Update noise varainces sigma2 (each for each dimension-column of the data matrix) 
    SW_sigma = vardist.gamma.*vardist.muW;
    SW2_sigma = vardist.gamma.*(vardist.muW.^2 + vardist.sigma2W); 
    t1 = sum(SW_sigma.*model.Likelihood.XtY', 2);
    t2 = sum(SW2_sigma.*repmat(diagXtX,Q,1), 2); 
    t3 = sum( (SW_sigma*XtXminusDiag).*SW_sigma, 2);
    if strcmp(model.Likelihood.noise, 'homosc') 
        model.Likelihood.sigma2 = sum( (model.Likelihood.YY  + (-2*t1 + t2 + t3)'))/(Q*N);
    else
        model.Likelihood.sigma2 = (model.Likelihood.YY - 2*t1' + t2' + t3')/N; 
        model.Likelihood.sigma2(model.Likelihood.sigma2<model.Likelihood.minSigma2)=model.Likelihood.minSigma2;
    end
    if debug==1
         F = slrLowerBound(model, vardist);
         if (F-Fold) < 0 
            fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (sigma2)\n',it, F, Fold,  F-Fold);
         end
         Fold = F;
    end
   
    % Update prior spike-and-slab probability alpha
    model.prior.alpha  = mean( vardist.gamma(:) );
    if debug==1
       F = slrLowerBound(model, vardist);
         if (F-Fold) < 0 
            fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (alpha)\n',it, F, Fold,  F-Fold);
       end
         Fold = F; 
    end
    
    % Update sigma2w (prior varaince over weights)
   model.prior.sigma2W = sum(sum(SW2_sigma ))./sum( vardist.gamma(:) );
    if debug==1
         F = slrLowerBound(model, vardist);
         if (F-Fold) < 0 
            fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (sigma2W)\n',it, F, Fold,  F-Fold);
         end
         Fold= F;
    end
    % M-STEP --- END
   
    
    % display the value of the lower bound if needed 
    if dispF==1
         F= slrLowerBound(model, vardist); 
         fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f\n',it, F, Fold,  F-Fold);
         Fold = F; 
    else
         if mod(it,dispEvery) == 0
             fprintf(1,'Iteration%4d/%4d\n',it,Iterations);
         end 
    end
    
    FF(it) = Fold; 
%    
end
    
if ~dispF
    FF = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = slrLowerBound(model, vdist)
% It compute the values of the lower bound for 
%
%

Q = model.Q; 
D = model.D; 
N = model.N;

gamma = vdist.gamma;
sigma2W = model.prior.sigma2W;
sigma2 = model.Likelihood.sigma2;
if length(sigma2) == 1
    sigma2 = sigma2*ones(1,Q);
end
alpha = model.prior.alpha;

%  
F1 = - (0.5*Q*N)*log(2*pi)... 
     - (0.5*N)*sum(log(sigma2),2)...
     - 0.5*sum(model.Likelihood.YY./sigma2,2);
 
F5 = - (0.5*D*Q)*log(2*pi*sigma2W) ...
        - 0.5*( sum(sum(1-gamma)) ...
        + sum(sum( gamma.*(vdist.muW.^2 + vdist.sigma2W), 2)'./sigma2W, 2));
     
F6 = log(alpha + (alpha==0))*sum(sum(gamma)) +  log(1-alpha+(alpha==1))*sum(sum(1 - gamma));

E1 = (0.5*D*Q)*log(2*pi*sigma2W) + 0.5*(D*Q) - 0.5*log(sigma2W)*sum(sum(gamma))...
     + 0.5*sum(sum( gamma.*log(vdist.sigma2W)  ));
 
E2 = - sum(sum( gamma.*log(gamma+(gamma==0)) + (1-gamma).*log(1-gamma+(gamma==1)) ));

SW_sigma  = gamma.*vdist.muW; 
diagXtX = model.Likelihood.diagXtX;
XtXminusDiag = model.Likelihood.XtXminusDiag;

F2  = sum(  sum(gamma.*vdist.muW.*model.Likelihood.XtY', 2)./sigma2'  ); 
F3 = sum(  sum(  (gamma.*(vdist.muW.^2 + vdist.sigma2W).*repmat(diagXtX, Q, 1)),2)./sigma2'  );
           
F4 = 0; 
for q=1:Q
   F4 = F4 + ( SW_sigma(q,:)*(XtXminusDiag*SW_sigma(q,:)' ))/sigma2(q);
end
F = F1 + F2 - 0.5*F3 - 0.5*F4 + F5 + F6 + E1 + E2;
