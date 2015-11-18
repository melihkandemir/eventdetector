function [out1, out2, out3] = varmkgpPredict(model, vardist, Xtest, Ytest)
%function [out1, out2, out3] = varmkgpPredict(model, vardist, Xtest, Ytest)
%
%
%  Michalis K. Titsias, 2011

M = model.M; 
Q = model.Q;
n = model.n;

if ~isempty(Xtest)
    nstar = size(Xtest, 1); 
else
    nstar = size(Ytest, 1); 
end

% compute the posterior q(phi_*m) = \int q(phi_*m |Phi_m) q(Phi_m) d Phi_m for m=1,...,M 
Phi_Mustar = zeros(nstar, M);
Phi_Sigmastar = ones(nstar, M);
for m=1:M
% 
    if ~strcmp(char(model.GP{m}.covfunc{1}), 'const') & ~strcmp(char(model.GP{m}.covfunc{1}), 'stadNormal')
         % compute the kernel matrix for the mth GP latent function 
         Km = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X);
         % cross covaraicne matrix with test inputs
         [Kss, Kstar] = feval(model.GP{m}.covfunc{:}, model.GP{m}.logtheta, model.X, Xtest);
    
         % Compute the covariance matrix of the variational 
         % distrbution q(phi_m), i.e. the matrix Sigma_phim 
         % (use matrix inversion lemma and numerically stable
         % computations) 
         Bm = eye(n) + vardist.alphaPhi(m)*Km;
         LBm = chol(Bm)';
         alpha = LBm'\(LBm\vardist.barmuPhi(:,m)); 
  
         Phi_Mustar(:,m) = Kstar'*alpha; 
         v = LBm\Kstar;
         Phi_Sigmastar(:, m) = Kss - vardist.alphaPhi(m)*sum(v.*v)'; 
    elseif strcmp(char(model.GP{m}.covfunc{1}), 'const')
         Phi_Mustar(:,m) = exp(model.GP{m}.logtheta); 
    end    
end

% Check is partially observed outputs are given. In such case  we need to re-update the 
% the variationla test posteriors Phi_Musta and Phi_Sigmastar to account for the information 
% given by the partially observed output data
if nargin == 4
      indexPresent = ones(size(Ytest)); 
      for q=1:model.Q
          indexMissing = isnan(Ytest(:, q));
          indexPresent(indexMissing,q) = 0;
      end
      indexPresent = sparse(indexPresent);

      sigma2 = model.Likelihood.sigma2;
      if size(sigma2,2) == 1
          sigma2 = sigma2*ones(1,Q);
      end
      SW_sigma  = (vardist.gamma.*vardist.muW)./repmat(sigma2', 1, M); 
      sqSW_sigma  = (vardist.gamma.*vardist.muW)./repmat(sqrt(sigma2)', 1, M); 
      SW2_sigma = (vardist.gamma.*(vardist.muW.^2 + vardist.sigma2W))./repmat(sigma2', 1, M); 

      % Do prediction separately for each test data point 
      for n=1:nstar
      %    
          % find firstly the posterior over the latent function values
          cond = find(indexPresent(n,:)==1);
          y = Ytest(n,cond); 
          barmu = y*SW_sigma(cond, :) - Phi_Mustar(n,:);
          Lambda = sqSW_sigma(cond, :)'*sqSW_sigma(cond, :);
          Lambda = Lambda - diag(diag(Lambda))  + diag(sum(SW2_sigma(cond,:),1));
          sqAlpha =  sqrt(Phi_Sigmastar(n,:))';
          Bm = eye(M) +sqAlpha*sqAlpha'.*Lambda;
          LBm = chol(Bm)'; 
          T = LBm\diag(sqAlpha);     
          
          % re-update the Phi_Mustar(n,:) and Phi_Sigmastar(n,:) 
          Phi_Sigmastar(n,:) = sum(T.*T)'; 
          Phi_Mustar(n,:) = (barmu*T')*T;
          %
      end
end

% Do the prediction
switch model.Likelihood.type 
    case 'Gaussian'
    %    
     
      % compute predicted means and variances which 
      % are provided analytically 
      out1 = zeros(nstar, Q); 
      out2 = zeros(nstar, Q); 
      
      SW_sigma =  vardist.gamma.*vardist.muW;
      SW2_sigma = vardist.gamma.*(vardist.muW.^2 + vardist.sigma2W);
      
      out1= Phi_Mustar*(SW_sigma');
      tmp1 = (Phi_Mustar.^2 + Phi_Sigmastar)*(SW2_sigma')... 
                    - (Phi_Mustar.^2)*(SW_sigma.^2)';
      if strcmp(model.Likelihood.noise, 'homosc') 
             out2 = tmp1 + model.Likelihood.sigma2;   
      else
             out2 = tmp1 + repmat(model.Likelihood.sigma2, nstar,1);   
      end          
           
      %for q=1:Q
      %%    
      %   SW_sigma = vardist.gamma(q, :).*vardist.muW(q, :);
      %   SW2_sigma = vardist.gamma(q, :).*(vardist.muW(q, :).^2 + vardist.sigma2W(q, :));
      %    
      %   out1(:, q) = sum(repmat( SW_sigma, nstar, 1).*Phi_Mustar, 2)'; 
      %              
      %   tmp = sum(repmat( SW2_sigma, nstar, 1).*(Phi_Mustar.^2 + Phi_Sigmastar), 2)...
      %         - sum( repmat( SW_sigma.^2, nstar, 1).*(Phi_Mustar.^2), 2);    
      %  if strcmp(model.Likelihood.noise, 'homosc') 
      %       out2(:,q) = tmp' + model.Likelihood.sigma2;   
      %  else
      %       out2(:,q) = tmp' + model.Likelihood.sigma2(q);   
      %  end
      %    
      %end
    %  
    case 'Probit'
    %     
      % number of samples in Monte Carlo 
      T = 100; 
      S = zeros(Q, M, T); 
      W = zeros(Q, M, T); 
      out1 = zeros(nstar, Q);
      for t=1:T  
      %    
          % draw firstly the independent samples 
          % from q(s_qm,w_qm)     
          S(:,:,t) = (rand(Q, M)<vardist.gamma); 
          W(:,:,t) = S(:,:,t).*(vardist.muW + sqrt(vardist.sigma2W).*randn(Q,M)) + (1-S(:,:,t)).*(model.prior.sigma2W*randn(Q,M)); 
          %W(:,:,t) = S(:,:,t).*(vardist.muW + sqrt(vardist.sigma2W).*randn(Q,M)) + (1-S(:,:,t)).*(repmat(model.prior.sigma2W',1,M).*randn(Q,M)); 
          %     
          for q=1:Q 
             Mustar = sum( repmat(S(q,:,t).*W(q,:,t), nstar, 1).*Phi_Mustar, 2);
             Sigma2star = sum( repmat(S(q,:,t).*(W(q,:,t).^2), nstar, 1).*Phi_Sigmastar, 2);
             out1(:,q) = out1(:,q) + probit( Mustar./sqrt(1 + Sigma2star) );
          end
      %    
      end
      out1 = out1/T;
    %
end

%sum(sum(abs(out1-out11)))
%sum(sum(abs(out2-out22)))
