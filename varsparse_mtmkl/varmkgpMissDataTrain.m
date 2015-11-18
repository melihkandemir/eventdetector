function [model vardist F] = varmkgpMissDataTrain(model, options, vardist)
%function [model vardist F] = varmkgpTrain(model, options, vardist)
%
%  INPUTS
%  model: is the modle structure 
%  options: severla optimizartion options (see demos)
%  varidist (optional): An alreayd initilaized/optimized variational distribution 
% 
%  OUTPUTS
%  model: updated model structured with learned hyperparameters
%  vardist: variational distribution 
%  F: final value of the variational lower bound 
%
%           
%    Michalis K. Titsias, 2011

%Jose C. Rubio 2014. Update: mult fails if model.indexPresent is not full
%model.indexPresent = full(model.indexPresent);

% make all NaN zeros
model.Y(~model.indexPresent) = 0;

Q = model.Q; 
M = model.M; 
n = model.n;

debug = 0; 
dispEvery = 1;
LearnKearnEvery = 5;

dispF = options(1); % display lower bound during running
kearnLearn = options(2); % learn kernel hyerparameters (0 for not learning) 
sigma2WLearn = options(3); % learn sigma2W hyperprameter (0 for not learning)
sigma2Learn = options(4); % learn likelihood noise parameters sigma2 (0 for not learning)
piLearn = options(5); % learn pi sparse mixing coefficient (0 for not learning)
sparsity = options(10); % use sparsity or not (if not pi is set to 1, is not learned)
Iterations = options(11); % number of variational EM iterations 


if nargin < 3      
        
% INITIALIZE FACTOR  q(s_qi=1) 
% (around 0.5)
gamma = 0.5*ones(Q, M) + 0.01*randn(Q, M); 
gamma(gamma<=0.2)=0.2;
gamma(gamma>=0.8)=0.8;
vardist.gamma = gamma;

% INITIALIZE FACTOR q(w_qm | s_qm =1) = N(w_qm| muW_qm,  sigma2_wqm)
% (close to the prior distribution)
%vardist.muW = assign+0.1*randn(Q, M) + model.prior.muW;
vardist.muW = sqrt(model.prior.sigma2W/M)*randn(Q, M) + model.prior.muW;
vardist.sigma2W = mean(model.prior.sigma2W)*ones(Q, M); 

% INITIALIZE FACTOR q(phi_m)
% (the variational distributions is fully specified by the alphas 
% and the variational means. To obtain the covariance matrices
% through the alphas we need to kernel matrices--- this only 
% only computed in demand and these big matrices are never stored 
% permantly into the model)
%

vardist.muPhi = zeros(n, M);
vardist.muPhiNoKm = zeros(n, M);
vardist.barmuPhi = zeros(n, M);
vardist.diagSigmaPhi = zeros(n, M);
vardist.alphaPhi = ones(n, M);
perm = randperm(Q); 
perm = repmat(perm,1,M);
for m=1:M
%    
  if ~strcmp(char(model.GP{m}.covfunc{1}), 'const')    
      if  ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')
          Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);      
          %SW2_sigma = (vardist.gamma(:, m).*(vardist.muW(:, m).^2 + vardist.sigma2W(:, m)))./model.Likelihood.sigma2'; 
          %vardist.alphaPhi(m) = 1;%sum(SW2_sigma, 1);
          Bm = eye(n) + Km;
          LBm = chol(Bm)';
          T = LBm\Km;
          vardist.diagSigmaPhi(:,m) =  diag(Km) - sum(T.*T)'; 
          vardist.barmuPhi(:, m) = 0*0.2*model.Y(:, perm(m)) +  randn(n,1); 
          vardist.muPhi(:, m)= T'*(LBm\vardist.barmuPhi(:, m));
      else
          vardist.alphaPhi(m) = 1;%sum(SW2_sigma, 1);
          vardist.diagSigmaPhi(:,m) =  1./2; 
          vardist.barmuPhi(:, m) = 0*0.2*model.Y(:, perm(m)) +  randn(n,1); 
          vardist.muPhi(:, m)= 2*randn(n,1); 
      end
  else
       vardist.muPhi(:, m) = exp(model.GP{m}.logtheta); 
       vardist.diagSigmaPhi(:,m) = 0;
  end
end



bias = zeros(1,Q);

% INITIALIZE FACTOR q(z_q) (truncated Gaussian for the multiple-output probit classification)
if strcmp(model.Likelihood.type, 'Probit') 
    vardist.meanZ = zeros(n, Q);
    imbFactor = size(1,Q);
    for q=1:Q
       % compute imbalance factor
       imbFactor(q) = size(find(model.Y(:,q)==1),1)/size(model.Y(:,q),1);
       % initialize so that to compensate for the imbalance of the classes
       vardist.meanZ(:,q) = abs(model.Y(:,q));%.*(model.Y(:,q)==1)*imbFactor(q) - model.Y(:,q).*(model.Y(:,q)==-1);
       %  have a fixed bias with the imbalanced factor
       bias(q) = norminv(imbFactor(q),0,1); 
    end      
    vardist.varZ = zeros(n, Q);
    vardist.entrZ = zeros(n, Q);
    model.imbFactor = imbFactor;
end
end

bias = zeros(1,Q);
model.bias = bias;


if sparsity == 0
    vardist.gamma = ones(Q, M);
end


 
Fold = -Inf;
% VARIATIONAL EM FOR MULTIPLE OUTPUT REGRESSION AND CLASSIFICATION 
switch model.Likelihood.type 
    case 'Gaussian'
        %
        % - Variational EM for multiple output regression 
        %
        for it=1:Iterations 
        % E-STEP --- START
       
           % quantities fixed at the E-STep 
           logPi = log(model.prior.pi/(1-model.prior.pi));
           sigma2Sigmaw = model.Likelihood.sigma2./model.prior.sigma2W;
           logSigma2Sigma2w = 0.5*log(sigma2Sigmaw);
           diagSigmaPhi = zeros(1,M);
           
           for m=1:M
           % 
           
           % indices exluding m
           setMinus = [1:m-1, (m+1):M];
           
           % compute the kernel matrix for the mth GP latent function  
           if ~strcmp(char(model.GP{m}.covfunc{1}), 'const') & ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')
           Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
           end
           
           % compute <phi_m^T phi_m> statistic 
           %phmTphim = vardist.muPhi(:,m)'*vardist.muPhi(:,m) + sum(vardist.diagSigmaPhi(:, m));
           phmTphim = vardist.muPhi(:,m).*vardist.muPhi(:,m) + vardist.diagSigmaPhi(:, m);
           phmTphim = phmTphim'*model.indexPresent;
           
           % compute  <phi_m^T phi_k> statistics for k 'not-equal' m  
           %phmTphik = sum( repmat(vardist.muPhi(:,m),[1, M-1]).*vardist.muPhi(:,setMinus), 1);
           phmTphik = repmat(vardist.muPhi(:,m),[1, M-1]).*vardist.muPhi(:,setMinus);
           phmTphik = phmTphik'*model.indexPresent;
                       
           % --- faster new code                 
           %b = (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))*phmTphik';
           b = sum( (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus)).*(phmTphik'), 2);
           
           diff = vardist.muPhi(:,m)'*model.Y - b';
           phmTphimSig = phmTphim + sigma2Sigmaw;
          
           u_qm = logPi + logSigma2Sigma2w - 0.5*log(phmTphimSig) ...
                       + (0.5./model.Likelihood.sigma2).*((diff.^2)./phmTphimSig);
           if sparsity == 1
               vardist.gamma(:, m) = 1./(1+exp(-u_qm'));                                % q(s_qm=1), q=1,...,Q          
           end
      
           vardist.muW(:, m) = (diff./phmTphimSig)';                                % q(w_qm | s_qm=1), q=1,...,Q
           vardist.sigma2W(:, m) = (model.Likelihood.sigma2./phmTphimSig)';
           % compute the bound after the update (this is for debugging)
           if debug==1
               Fnew = varmkgpLowerBound(model, vardist);
               if  (Fnew-Fold) < 0 
                 fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(s,w))\n',it, Fnew, Fold,  Fnew-Fold);
               end
               Fold = Fnew;
           end   
         
           % update factor q(phi_m)
           if ~strcmp(char(model.GP{m}.covfunc{1}), 'const')      
           SW_sigma = (vardist.gamma(:, m).*vardist.muW(:, m))./model.Likelihood.sigma2'; 
           SW2_sigma = (vardist.gamma(:, m).*(vardist.muW(:, m).^2 + vardist.sigma2W(:, m)))./model.Likelihood.sigma2'; 
           % --- old code for b --- 
           %b = zeros(n,1);
           %for kk=1:M-1
           %    k = setMinus(kk);  
           %    b = b + sum(  SW_sigma.*(vardist.gamma(:, k).*vardist.muW(:, k)), 1)*vardist.muPhi(:,k);
           %end
           % --- new code for b --- 
           %b = (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))'*SW_sigma;
           %b = vardist.muPhi(:,setMinus)*b;  
          
           b = vardist.muPhi(:,setMinus)*(vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))';
           b(~model.indexPresent)=0;
           b = b*SW_sigma;
           
           % update factor q(phi_m)
           %vardist.alphaPhi(m) = sum(SW2_sigma, 1);
           vardist.alphaPhi(:,m) = full(model.indexPresent)*SW2_sigma;
           %barmuPhi = sum( repmat(SW_sigma', n, 1).*model.Y, 2) - b;
           barmuPhi = model.Y*SW_sigma - b;
           
           vardist.barmuPhi(:, m) = barmuPhi;
           if  ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')
                sqAlpha = sqrt(vardist.alphaPhi(:,m));
                Bm = eye(n) +sqAlpha*sqAlpha'.*Km;
                LBm = chol(Bm)';
                T = LBm\(repmat(sqAlpha,1,model.n).*Km);     
                vardist.diagSigmaPhi(:,m) = diag(Km) - sum(T.*T)';
                vardist.muPhiNoKm(:, m) = barmuPhi  - sqAlpha.*(LBm'\(LBm\(sqAlpha.*(Km*barmuPhi))));   
                vardist.muPhi(:, m) = Km*vardist.muPhiNoKm(:, m); 
           else
                vardist.diagSigmaPhi(:,m) = 1./(1 + vardist.alphaPhi(:,m)); 
                vardist.muPhi(:, m) = barmuPhi./(1 + vardist.alphaPhi(:,m));
           end
           end
           
           % compute the bound after the update (this is for debugging)
           if debug==1
               Fnew = varmkgpLowerBound(model, vardist);
               if (Fnew-Fold) < 0 
               fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(phi_m))\n',it, Fnew, Fold,  Fnew-Fold);
               end
               Fold = Fnew;
           end
       
           % keep some auxiliary variables used in the M-step 
           diagSigmaPhi(m) = sum(vardist.diagSigmaPhi(:,m));
           
           end
      
           
           % E-STEP --- END
        
           % M-STEP --- START 
         
           % 1) noise likelihood parameters sigma_q^2
           if sigma2Learn == 1
           SW_sigma  = vardist.gamma.*vardist.muW;
           SW2_sigma = vardist.gamma.*(vardist.muW.^2 + vardist.sigma2W); 
           %muPhiTmuPhi = vardist.muPhi'*vardist.muPhi;
           muPhiTmuPhi = vardist.muPhi.*vardist.muPhi  + vardist.diagSigmaPhi;
           muPhiTmuPhi = muPhiTmuPhi'*model.indexPresent;
           
           %muPhiTmuPhi0 = muPhiTmuPhi - diag(diag(muPhiTmuPhi));
           t1 = sum(SW_sigma.*(model.Y'*vardist.muPhi), 2);
           %t2 = sum(SW2_sigma.*repmat(diag(muPhiTmuPhi)' + diagSigmaPhi,Q,1), 2); 
           t2 = sum(SW2_sigma'.* muPhiTmuPhi,1); 
           %t3 = sum( (SW_sigma*muPhiTmuPhi0).*SW_sigma, 2);
           
           t3 = zeros(1, model.Q);
           for m =1:M
                for m1=m+1:M
                %    
                    tt = ( (vardist.gamma(:, m1).*vardist.muW(:, m1)).*SW_sigma(:, m) )';
                    t3 = t3 + tt.*((vardist.muPhi(:,m1).*vardist.muPhi(:,m))'*model.indexPresent); 
                %    
                end
           end
             
           if strcmp(model.Likelihood.noise, 'homosc') 
               %model.Likelihood.sigma2 = sum( (model.Likelihood.YY  + (-2*t1 + t2 + t3)'))/(Q*n); 
               model.Likelihood.sigma2 = sum( (model.Likelihood.YY  + (-2*t1' + t2 + 2*t3)))/sum(model.numPresent); 
           else    
               %model.Likelihood.sigma2 = (model.Likelihood.YY  + (-2*t1 + t2 + t3)')/n;
               model.Likelihood.sigma2 = (model.Likelihood.YY  + (-2*t1' + t2 + 2*t3))./model.numPresent;
           end
        
           model.Likelihood.sigma2(model.Likelihood.sigma2<model.Likelihood.minSigma2)=model.Likelihood.minSigma2;
           if debug==1
                  Fnew = varmkgpLowerBound(model, vardist);
                  if (Fnew-Fold) < 0 
                    fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (sigma2)\n',it, Fnew, Fold,  Fnew-Fold);
                  end
                  Fold = Fnew;
           end
           end
         
           % 2) sparse prior pi 
           if piLearn == 1
           model.prior.pi = mean(vardist.gamma(:)); 
           if debug==1
              Fnew = varmkgpLowerBound(model, vardist);
              if (Fnew-Fold) < 0 
                 fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (pi)\n',it, Fnew, Fold,  Fnew-Fold);
              end
              Fold = Fnew; 
           end
           end
           
           % 3) variance for the weigths sigma2W
           if sigma2WLearn == 1
           
           model.prior.sigma2W = sum(sum( vardist.gamma.*(vardist.muW.^2+vardist.sigma2W) ))/sum(sum(vardist.gamma));
           %sumGamma = sum(vardist.gamma, 2)';
           %sumGamma = sumGamma + (sumGamma==0); 
           %model.prior.sigma2W = (sum( vardist.gamma.*(vardist.muW.^2+vardist.sigma2W), 2)')./sumGamma;
           
           if debug==1
              Fnew = varmkgpLowerBound(model, vardist);
              if (Fnew-Fold) < 0 
                 fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (sigma2W)\n',it, Fnew, Fold,  Fnew-Fold);
              end
              Fold = Fnew; 
           end
           end
           
           % 4) update also kernel hyperparameters if needed 
           % (only every 10 iteration perform this very expensive step)
           if kearnLearn == 1 & mod(it,LearnKearnEvery)==0     
             for m=1:M  
               % only if there exist kernel hyperparameters  
               if (model.GP{m}.nParams > 0) & (~strcmp(char(model.GP{m}.covfunc{1}), 'const')  & ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')) & (vardist.alphaPhi(m) > 1e-06) 
               fprintf(1,'Iteration%4d  M-Step for kernel %4d\n',it,m);
               setMinus = [1:m-1, (m+1):M];
               SW_sigma  = (vardist.gamma(:, m).*vardist.muW(:, m))./model.Likelihood.sigma2'; 
               SW2_sigma = (vardist.gamma(:, m).*(vardist.muW(:, m).^2 + vardist.sigma2W(:, m)))./model.Likelihood.sigma2'; 
               
               b = vardist.muPhi(:,setMinus)*(vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))';
               b(~model.indexPresent)=0;
               b = b*SW_sigma;
               barmuPhi = model.Y*SW_sigma - b;   
               alphaPhi = full(model.indexPresent)*SW2_sigma;   
               ybar = barmuPhi./alphaPhi;
               
               ycovfunc = model.GP{m}.covfunc;
               logtheta = model.GP{m}.logtheta(:)';
             
               [logtheta fX] = minimize(logtheta(:), 'gpr_fnNew', 5, ycovfunc, model.X, ybar, 1./alphaPhi);
               model.GP{m}.logtheta = logtheta(:)'; 
               
               % Do also a full update of the factor q(phi_m)
               Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
               vardist.alphaPhi(:,m) = full(model.indexPresent)*SW2_sigma;
               sqAlpha =  sqrt(vardist.alphaPhi(:,m));
               Bm = eye(n) +sqAlpha*sqAlpha'.*Km;
               LBm = chol(Bm)'; 
               T = LBm\(repmat(sqAlpha,1,model.n).*Km);     
               vardist.diagSigmaPhi(:,m) =  diag(Km) - sum(T.*T)'; 
               barmuPhi = model.Y*SW_sigma - b;      
               vardist.barmuPhi(:, m) = barmuPhi;
               vardist.muPhiNoKm(:, m) = barmuPhi  - sqAlpha.*(LBm'\(LBm\(sqAlpha.*(Km*barmuPhi))));   
               vardist.muPhi(:, m) = Km*vardist.muPhiNoKm(:, m);   
               if debug==1
                  Fnew = varmkgpLowerBound(model, vardist);
                  if (Fnew-Fold) < 0 
                     fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(phi_m), kernel)\n',it, Fnew, Fold,  Fnew-Fold);
                  end
                  Fold = Fnew; 
               end
               end
             end
           end
           % M-STEP --- END
        
           % print the lower bound 
           if dispF == 1
              Fnew = varmkgpLowerBound(model, vardist);
              fprintf(1,'Iteration%4d/%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f\n',it,Iterations, Fnew, Fold,  Fnew-Fold);
              Fold = Fnew; 
           else
              if mod(it,dispEvery) == 0
                fprintf(1,'Iteration%4d/%4d\n',it,Iterations);
              end
           end
        
        end % iteration loop
    %    
    case 'Probit'
        %  
        % - Variational EM for multiple output classification  
        %
        
        for it=1:Iterations 
        % E-STEP --- START
           % quantities fixed at the E-Step 
           logPi = log(model.prior.pi/(1-model.prior.pi));
           sigma2Sigmaw = model.Likelihood.sigma2./model.prior.sigma2W;
           logSigma2Sigma2w = 0.5*log(sigma2Sigmaw);
           diagSigmaPhi = zeros(1,M);
        
           % KEEP FIXED TRUNCATED GAUSSIAN AND MAXIMIZE 
           % (you simply need to define regression data y_q * < z_q >
           YmeanZ = model.Y.*vardist.meanZ;
           perm = randperm(M);
           for mm=1:M
           m = perm(mm);    
            
           % indices exluding m
           setMinus = [1:m-1, (m+1):M];         
           
           % compute the kernel matrix for the mth GP latent function 
           if ~strcmp(char(model.GP{m}.covfunc{1}), 'const')
           Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
           end
            
           % compute <phi_m^T phi_m> statistic 
           phmTphim = vardist.muPhi(:,m)'*vardist.muPhi(:,m) + sum(vardist.diagSigmaPhi(:, m));
         
           % compute  <phi_m^T phi_k> statistics for k 'not-equal' m  
           phmTphik = sum( repmat(vardist.muPhi(:,m),[1, M-1]).*vardist.muPhi(:,setMinus), 1);
              
           % --- faster new code
           b = (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))*phmTphik';
           diff = vardist.muPhi(:,m)'*YmeanZ - b' - bias*sum(vardist.muPhi(:,m),1); 
           phmTphimSig = phmTphim + sigma2Sigmaw;
           u_qm = logPi + logSigma2Sigma2w - 0.5*log(phmTphimSig) ...
                       + (0.5./model.Likelihood.sigma2).*((diff.^2)./phmTphimSig);
           if sparsity == 1 
              vardist.gamma(:, m) = 1./(1+exp(-u_qm'));                                % q(s_qm=1), q=1,...,Q            
           end
           vardist.muW(:, m) = (diff./phmTphimSig)';                                % q(w_qm | s_qm=1), q=1,...,Q
           vardist.sigma2W(:, m) = (model.Likelihood.sigma2./phmTphimSig)'; 
           % compute the bound after the update (this is for debugging)
           if debug==1
               mm = model;
               mm.Y = model.Y.*vardist.meanZ;
               mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
               Fnew = varmkgpLowerBound(mm, vardist);
               % put term from the truncated Gaussian
               Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));            
               if  (Fnew-Fold) < 0 
                 fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(s,w))\n',it, Fnew, Fold,  Fnew-Fold);
               end
               Fold = Fnew;
           end
           end
           
           % update factor q(phi_m)
           perm = randperm(M);
           for mm=1:M
           m = perm(mm);      
           if ~strcmp(char(model.GP{m}.covfunc{1}), 'const')  
           Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
           % indices exluding m
           setMinus = [1:m-1, (m+1):M];
           SW_sigma  = (vardist.gamma(:, m).*vardist.muW(:, m))./model.Likelihood.sigma2'; 
           SW2_sigma = (vardist.gamma(:, m).*(vardist.muW(:, m).^2 + vardist.sigma2W(:, m)))./model.Likelihood.sigma2'; 
           % --- new code for b --- 
           b = (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))'*SW_sigma;
           b = vardist.muPhi(:,setMinus)*b;  
           % update factor q(phi_m)
           % --- old code 
           %vardist.alphaPhi(m) = sum(SW2_sigma, 1);
           %Bm = eye(n) + vardist.alphaPhi(m)*Km;
           %LBm = chol(Bm)';
           %T = LBm\Km; 
           %SigmaPhim = Km - vardist.alphaPhi(m)*(T'*T); 
           %barmuPhi = sum( repmat(SW_sigma', n, 1).*YmeanZ, 2) - b;
           %vardist.barmuPhi(:, m) = barmuPhi;
           %vardist.muPhi(:, m)= SigmaPhim*barmuPhi;            
           %vardist.diagSigmaPhi(:,m) = diag(SigmaPhim);
           % --- new code
           vardist.alphaPhi(m) = sum(SW2_sigma, 1);
           Bm = eye(n) + vardist.alphaPhi(m)*Km;
           LBm = chol(Bm)';
           T = LBm\Km; 
           vardist.diagSigmaPhi(:,m) =  diag(Km) - vardist.alphaPhi(m)*sum(T.*T)'; 
           barmuPhi = sum( repmat(SW_sigma', n, 1).*YmeanZ, 2) - b - bias*SW_sigma;
           vardist.barmuPhi(:, m) = barmuPhi;
           vardist.muPhi(:, m) = T'*(LBm\barmuPhi);
           %vardist.muPhi(:, m) = Km*(LBm'\(LBm\barmuPhi));
           end
           % compute the bound after the update (this is for debugging)
           if debug==1
               mm = model;
               mm.Y = model.Y.*vardist.meanZ;
               mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
               Fnew = varmkgpLowerBound(mm, vardist);
               % put term from the truncated Gaussian
               Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));            
               if (Fnew-Fold) < 0 
               fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(phi_m))\n',it, Fnew, Fold,  Fnew-Fold);
               end
               Fold = Fnew;
           end
       
           % keep some auxiliary variables used in the M-step 
           diagSigmaPhi(m) = sum(vardist.diagSigmaPhi(:,m));
           
           end % end for loop M
        
           % update the truncated Gaussians    
           for q=1:Q   
              MeanSumPhim = model.Y(:,q).*(bias(q) + sum( repmat( vardist.gamma(q, :).*vardist.muW(q, :), n, 1).*vardist.muPhi, 2));
              % update the mean of the truncated Gaussian 
              [vardist.meanZ(:, q), vardist.varZ(:, q), vardist.entrZ(:, q)]  = truncNormalStats(MeanSumPhim);
              if debug==1
                 mm = model;
                 mm.Y = model.Y.*vardist.meanZ;
                 mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
                 Fnew = varmkgpLowerBound(mm, vardist);
                 % put term from the truncated Gaussian
                 Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));   
                 if (Fnew-Fold) < 0 
                     fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(z_q))\n',it, Fnew, Fold,  Fnew-Fold);
                 end
                 Fold = Fnew; 
              end
           end
           % E-STEP --- END
        
        
           % M-STEP --- START 
           %
           % 1) sparse prior pi
           if piLearn == 1
           model.prior.pi = mean(vardist.gamma(:)); 
           if debug==1
               mm = model;
               mm.Y = model.Y.*vardist.meanZ;
               mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
               Fnew = varmkgpLowerBound(mm, vardist);
               % put term from the truncated Gaussian
               Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));   
               if (Fnew-Fold) < 0 
                  fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (pi)\n',it, Fnew, Fold,  Fnew-Fold);
               end
               Fold = Fnew; 
           end
           end
           
           
           % 3) variance for the weights sigma2W
           if sigma2WLearn == 1
           model.prior.sigma2W = sum(sum( vardist.gamma.*(vardist.muW.^2+vardist.sigma2W) ))/sum(sum(vardist.gamma));
           %model.prior.sigma2W = model.prior.sigma2W*ones(1,Q);
           %sumGamma = sum(vardist.gamma, 2)';
           %sumGamma = sumGamma + (sumGamma==0); 
           %model.prior.sigma2W = (sum( vardist.gamma.*(vardist.muW.^2+vardist.sigma2W), 2)')./sumGamma;
           if debug==1
               mm = model;
               mm.Y = model.Y.*vardist.meanZ;
               mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
               Fnew = varmkgpLowerBound(mm, vardist);
               % put term from the truncated Gaussian
               Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));   
               if (Fnew-Fold) < 0 
                 fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (sigma2W)\n',it, Fnew, Fold,  Fnew-Fold);
               end
               Fold = Fnew; 
           end
           end
          
           
           % update also kernel hyperparameters if needed 
           % (only every 10 iteration perform this very expensive step)
           if kearnLearn == 1 & mod(it,LearnKearnEvery)==0
             YmeanZ = model.Y.*vardist.meanZ; 
             for m=1:M  
               % only if there exist kernel hyperparameters  
               if (model.GP{m}.nParams > 0) & (~strcmp(char(model.GP{m}.covfunc{1}), 'const')) & (vardist.alphaPhi(m) > 1e-06)
               fprintf(1,'Iteration%4d  M-Step for kernel %4d\n',it,m);
               setMinus = [1:m-1, (m+1):M];
               SW_sigma  = (vardist.gamma(:, m).*vardist.muW(:, m))./model.Likelihood.sigma2'; 
               SW2_sigma = (vardist.gamma(:, m).*(vardist.muW(:, m).^2 + vardist.sigma2W(:, m)))./model.Likelihood.sigma2'; 
               b = (vardist.gamma(:, setMinus).*vardist.muW(:, setMinus))'*SW_sigma;
               b = vardist.muPhi(:, setMinus)*b;             
               barmuPhi = sum( repmat(SW_sigma', n, 1).*YmeanZ, 2) - b - bias*SW_sigma;
               alphaPhi = sum(SW2_sigma, 1);
               ybar = barmuPhi/alphaPhi;
               ycovfunc = {'covSum', {model.GP{m}.covfunc,'covNoise'}};
               logtheta = [model.GP{m}.logtheta(:)', -0.5*log(alphaPhi)];
               [logtheta fX] = minimize(logtheta(:), 'gpr_fn', 5, ycovfunc, model.X, ybar);
               logtheta = logtheta(1:end-1);
               model.GP{m}.logtheta = logtheta(:)'; 
               
               % Do also a full update of the factor q(phi_m)
               Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
               vardist.alphaPhi(m) = sum(SW2_sigma, 1);
               Bm = eye(n) + vardist.alphaPhi(m)*Km;
               LBm = chol(Bm)';
               T = LBm\Km; 
               vardist.diagSigmaPhi(:,m) = diag(Km) - vardist.alphaPhi(m)*sum(T.*T)'; 
               barmuPhi = sum( repmat(SW_sigma', n, 1).*YmeanZ, 2) - b - bias*SW_sigma;
               vardist.barmuPhi(:, m) = barmuPhi;
               vardist.muPhi(:, m) = T'*(LBm\barmuPhi);
               
               if debug==1
                  mm = model;
                  mm.Y = model.Y.*vardist.meanZ;
                  mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
                  Fnew = varmkgpLowerBound(mm, vardist);
                  % put term from the truncated Gaussian
                  Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));   
                  if (Fnew-Fold) < 0 
                     fprintf(1,'Iteration%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f (q(phi_m), kernel)\n',it, Fnew, Fold,  Fnew-Fold);
                  end
                  Fold = Fnew; 
               end
               end
             end
           end
           %
           % M-STEP --- END
        
           % print the lower bound 
           if dispF == 1
               mm = model;
               mm.Y = model.Y.*vardist.meanZ;
               mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
               Fnew = varmkgpLowerBound(mm, vardist); 
               % put term from the truncated Gaussian
               Fnew = Fnew - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));   
               fprintf(1,'Iteration%4d/%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f\n',it,Iterations, Fnew, Fold,  Fnew-Fold);
               Fold = Fnew;
           else
               if mod(it,dispEvery) == 0
                  fprintf(1,'Iteration%4d/%4d\n',it,Iterations);
               end
           end
        
        end % iterations loop   

end % switch closes


% return final value of the lower bound 
if strcmp(model.Likelihood.type, 'Gaussian')
    F = varmkgpLowerBound(model, vardist);
elseif strcmp(model.Likelihood.type, 'Probit')
    mm = model;
    mm.Y = vardist.meanZ; 
    mm.Likelihood.YY = sum(mm.Y.*mm.Y,1);
    F = varmkgpLowerBound(mm, vardist);
    % put term from the truncated Gaussian
    F = F - 0.5*sum(vardist.varZ(:)) + sum(vardist.entrZ(:));   
end
    

%  AUXILIARY FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = varmkgpLowerBound(model, vdist)
% It compute the values of the lower bound for 
%
%

Q = model.Q; 
M = model.M; 
n = model.n;
gamma = vdist.gamma;
sigma2W = model.prior.sigma2W;
sigma2 = model.Likelihood.sigma2;
if size(sigma2,2) == 1
    sigma2 = sigma2*ones(1,Q);
end
Ppi = model.prior.pi;

%  
F1 = - (0.5*sum(model.numPresent))*log(2*pi)... 
     - 0.5*sum(model.numPresent.*log(sigma2),2)...
     - 0.5*sum(model.Likelihood.YY./sigma2,2);
 
F5 = - (0.5*M*Q)*log(2*pi) - (0.5*M)*sum(log(sigma2W),2)...
     - 0.5*( sum(sum(1-gamma)) ...
     + sum(sum( gamma.*(vdist.muW.^2 + vdist.sigma2W), 2)'./sigma2W, 2));
     
F6 = log(Ppi + (Ppi==0))*sum(sum(gamma)) +  log(1-Ppi+(Ppi==1))*sum(sum(1 - gamma));

E1 = (0.5*M*Q)*log(2*pi) + (0.5*M)*sum(log(sigma2W)) + 0.5*(M*Q) - 0.5*sum(log(sigma2W).*(sum(gamma,2)'))...
     + 0.5*sum(sum( gamma.*log(vdist.sigma2W)  ));
 
E2 = - sum(sum( gamma.*log(gamma+(gamma==0)) + (1-gamma).*log(1-gamma+(gamma==1)) ));

%PP = sum(vdist.muPhi.*vdist.muPhi, 1);
%phimTphim = zeros(1,M);

SW_sigma  = (gamma.*vdist.muW)./repmat(sigma2', 1, M); 
SW2_sigma = (gamma.*(vdist.muW.^2 + vdist.sigma2W))./repmat(sigma2', 1, M); 

if model.existBias == 1
  F7PlusE3 = 0.5*((M-1)*n);
else
  F7PlusE3 = 0.5*(M*n);
end

%F2 = sum(sum( repmat(model.bias./sigma2, n, 1).*model.Y, 2)); 

F3 = sum((model.bias.^2)./sigma2, 2)*n; 
rPhi = zeros(n,1);
for m=1:M
  %    
     tt = (gamma(:, m).*vdist.muW(:, m)).*((model.bias./sigma2)');
     rPhi = rPhi +  sum(tt,1)*vdist.muPhi(:,m); 
  %    
  end
  
 F4 = sum(rPhi);
 
%
for m=1:M
%
  % compute the kernel matrix for the mth GP latent
  % function 
  if ~strcmp(char(model.GP{m}.covfunc{1}), 'const') 
      if ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')
          Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
          % Compute the covariance matrix of the variational 
          % distribution q(phi_m), i.e. the matrix Sigma_phim 
          % (use matrix inversion lemma and numerically stable
          % computations)    
          sqAlpha = sqrt(vdist.alphaPhi(:,m));
          sqAKmsqA = sqAlpha*sqAlpha'.*Km;
          Bm = eye(n)  + sqAKmsqA;
          LBm = chol(Bm)';
          invLBm = LBm\eye(n);
          invBm = invLBm'*invLBm;
          T = LBm\(repmat(sqAlpha,1,model.n).*Km);      
          diagSigmaPhim =  diag(Km) - sum(T.*T)';  
      else
          diagSigmaPhim = 1./(1+vdist.alphaPhi(:,m)); 
      end
  end

  %F2 = F2 + sum( repmat(SW_sigma(:, m)', n, 1).*model.Y, 2)'*vdist.muPhi(:,m);  
  F3 = F3 + sum( (full(model.indexPresent)*SW2_sigma(:, m)).*(vdist.muPhi(:,m).*vdist.muPhi(:,m) + diagSigmaPhim) );
  %F3 = F3 + sum( SW2_sigma(:, m), 1)*phimTphim(m);

  % 
  rPhi = zeros(n,1); 
  for m1=m+1:M
  %    
     tt = (gamma(:, m1).*vdist.muW(:, m1)).*SW_sigma(:, m);
     tt = full(model.indexPresent)*tt; 
     rPhi = rPhi + tt.*vdist.muPhi(:,m1); 
     %rPhi = rPhi +  sum(tt,1)*vdist.muPhi(:,m1); 
  %    
  end
  
  F4 = F4 + rPhi'*vdist.muPhi(:,m);
  
  % there no prior/KL term for the const function 
  if ~strcmp(char(model.GP{m}.covfunc{1}), 'const')  
      if ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')
           F7PlusE3 = F7PlusE3 - 0.5*n - sum(log(diag(LBm)))  + 0.5*sum(sum(invBm.*sqAKmsqA)) ...
                            - 0.5*vdist.muPhiNoKm(:, m)'*(Km*vdist.muPhiNoKm(:, m));
           %invBm_barmuPhi = invBm*vdist.barmuPhi(:, m); 
           %F7PlusE311 = F7PlusE311 - sum(log(diag(LBm)))  - 0.5*sum(diag(invBm)) ...
           %                 - 0.5*invBm_barmuPhi'*(Km*invBm_barmuPhi);            
      else
           invBm_barmuPhi = vdist.barmuPhi(:, m)./(1+vdist.alphaPhi(:,m)); 
           F7PlusE3 = F7PlusE3 - 0.5*sum(log(1+vdist.alphaPhi(:,m)))  - 0.5*sum(1./(1+vdist.alphaPhi(:,m))) ...
                            - 0.5*invBm_barmuPhi'*invBm_barmuPhi;   
      end
  end
  %
end
F2 = sum(sum( SW_sigma.*(model.Y'*vdist.muPhi) ));

%fprintf(1,'F1234 %11.6f\n', F1+F2+F3+F4);
%fprintf(1,'F1 %11.6f\n', F1);
%fprintf(1,'F2 %11.6f\n', F2);
%fprintf(1,'F3 %11.6f\n', F3);
%fprintf(1,'F4 %11.6f\n', F4);
%fprintf(1,'F5 %11.6f\n', F5);
%fprintf(1,'F6 %11.6f\n', F6);
%fprintf(1,'E1 %11.6f\n', E1);
%fprintf(1,'E2 %11.6f\n', E2);
%fprintf(1,'F7PlusE3 %11.6f\n', F7PlusE3);
%pause

F = F1 + F2 - 0.5*F3 - F4 + F5 + F6 + E1 + E2 + F7PlusE3; 

% fro DEbugging -- only the variational bound for the sigma_q^2
function F = varmkgpLowerBoundSq(model, vdist, q)
% It compute the values of the lower bound for 
%
%

Q = model.Q; 
M = model.M; 
n = model.n;
gamma = vdist.gamma(q, :);
sigma2W = model.prior.sigma2W;
sigma2 = model.Likelihood.sigma2(q);
Ppi = model.prior.pi;

%  
F1 = - (0.5*n)*log(2*pi)... 
     - (0.5*n)*sum(log(sigma2))...
     - 0.5*(model.Likelihood.YY(q)/sigma2);
 

PP = sum(vdist.muPhi.*vdist.muPhi, 1);
phimTphim = zeros(1,M);
F2 = 0; F3 = 0; F4 = 0;

SW_sigma  = (gamma.*vdist.muW(q,:))/sigma2; 
SW2_sigma = (gamma.*(vdist.muW(q,:).^2 + vdist.sigma2W(q,:)))/sigma2; 

%
for m=1:M
%
  % compute the kernel matrix for the mth GP latent
  % function 
  Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
 
  % Compute the covariance matrix of the variational 
  % distribution q(phi_m), i.e. the matrix Sigma_phim 
  % (use matrix inversion lemma and numerically stable
  % computations) 
  Bm = eye(n) + vdist.alphaPhi(m)*Km;
  LBm = chol(Bm)';
  invLBm = LBm\eye(n);
  invBm = invLBm'*invLBm;
  T = invLBm*Km; 
  SigmaPhim = Km - vdist.alphaPhi(m)*(T'*T); 
  % compute <phi_m^T phi_m>, m=1,...,M 
  phimTphim(m) = PP(m) + sum(diag(SigmaPhim));
  
  F2 = F2 + (SW_sigma(m)*model.Y(:,q))'*vdist.muPhi(:,m); 
  F3 = F3 + SW2_sigma(m)*phimTphim(m);
  
  % 
  rPhi = zeros(n,1);
  for m1=m+1:M
  %    
     tt = (gamma(m1)*vdist.muW(q, m1))*SW_sigma(m);
     rPhi = rPhi +  tt*vdist.muPhi(:,m1); 
  %    
  end
  
  F4 = F4 + rPhi'*vdist.muPhi(:,m);
  %
end
  
F = F1 + F2 - 0.5*F3 - F4; 


% fro DEbugging -- only the variational bound for the z_q
function F = varmkgpLowerBoundZq(model, vdist, q)
% It compute the values of the lower bound for 
%
%

M = model.M; 
gamma = vdist.gamma(q, :);
sigma2 = model.Likelihood.sigma2(q);

F1 = - 0.5*(sum(vdist.meanZ(:,q).*vdist.meanZ(:,q),1)/sigma2);
F2 = 0;
SW_sigma  = (gamma.*vdist.muW(q,:))/sigma2; 
for m=1:M
  F2 = F2 + (SW_sigma(m)*vdist.meanZ(:,q))'*(model.Y(:,q).*vdist.muPhi(:,m)); 
end
F = F1 + F2 - 0.5*sum(vdist.varZ(:, q)) + sum(vdist.entrZ(:, q));   
