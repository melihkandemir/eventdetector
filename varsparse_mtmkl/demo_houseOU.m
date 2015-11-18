% Demo for house denoising
clear all;
addpath misc_toolbox/;
addpath misc_toolbox/gplm/;


for noiselevel = [15 25 50]
    
close all;

train = 1; % Set to 0 if you want to load and display results

% Fix seeds
randn('seed', 11);
rand('seed', 11);

% number of GP latent GP functions 
M = 64;
Iters = 60; 

% load data
picorig = double(imread('house.png'));
[Nh, Nw] = size(picorig);
pic = picorig + noiselevel*randn(Nh,Nw);

PSNRpic = 20*log10(255/sqrt(mean(mean((pic-picorig).^2))))

image(pic)
colormap((0:255)'/255*[1 1 1])

B=8; % block size is BxB
h = (1:B)'*ones(1,B);
w = ones(B,1)*(1:B);
X = [h(:) w(:)];
[n, D]=size(X);

% Cut in overlapping blocks
Nx = Nh-B+1;
Ny = Nw-B+1;
Y = zeros(n, Nx*Ny);
for h = 1:Nx
    for w = 1:Ny
        block = pic(h:h+B-1,w:w+B-1);
        Y(:,(h-1)*Ny+w) = block(:);
        
    end
end

kernels={};
for m=1:M 
    kernels{m} = {'covOUiso'};
end


noise = 'homosc'; % anything else will mean ouput-specific (heteroscedastic) noise 

% substruct the mean and scale the data 
model = varmkgpCreate(X, (Y - repmat(mean(Y),size(Y,1),1))/sqrt(var(Y(:))), 'Gaussian', kernels, noise);
% keep the substracted mean and scale for prediction purposes
model.meanY = mean(Y);
model.scale = sqrt(var(Y(:)));
clear Y
model.Likelihood.sigma2=1; % better initial guess

% training 
% load default options 
load defoptions;
options(1) = 1; % display lower bound during running...
options(2) = 1; % learn kernel hyerparameters (0 for not learning)... 
options(3) = 1; % learn sigma2W hyperprameter (0 for not learningg)...
options(4) = 1; % learn likelihood noise parameters sigma2 (0 for not learning)...
options(5) = 1; % learn pi sparse mixing coefficient (0 for not learning)...
options(10) = 1 ; % use sparsity or not (if not pi is set to 1, is not learned)..
options(11) = Iters; % number of variational EM iterations;

if train == 1
    [model vardist margLogL] = varmkgpTrain(model, options);
    save(['house_resultOU_' num2str(noiselevel)])
else
    load(['house_resultOU_' num2str(noiselevel)])
end

% mean reconstruction of the training data
predF = vardist.muPhi*((vardist.gamma.*vardist.muW)');
predF = predF*model.scale+ones(n,1)*model.meanY;

num = zeros(Nh,Nw);
den = zeros(Nh,Nw);
for h = 1:Nx
    for w = 1:Ny
        num(h:h+B-1,w:w+B-1) = num(h:h+B-1,w:w+B-1) + reshape(predF(:,(h-1)*Ny+w),B,B);
        den(h:h+B-1,w:w+B-1) = den(h:h+B-1,w:w+B-1) +1;
    end
end
picest = num./den;

PSNRpicest = 20*log10(255/sqrt(mean(mean((picest-picorig).^2))))

close all
image(pic);axis image;axis off;colormap((0:255)'/255*[1 1 1])
print -depsc  ./NIPS/figs/houseOUnoisy
image(picest);axis image;axis off;colormap((0:255)'/255*[1 1 1])
print -depsc  ./NIPS/figs/houseOUdenoised

subplot(2,2,1);image(picorig);axis image
subplot(2,2,2);image(pic);xlabel(sprintf('PSNR %.2f dB',PSNRpic)); axis image
subplot(2,2,3);image(picest); xlabel(sprintf('PSNR %.2f dB',PSNRpicest)); axis image

% Importance order
err = zeros(M,1);
for m = 1:M
    err(m) = norm(model.scale*vardist.muPhi(:,m)*((vardist.gamma(:,m).*vardist.muW(:,m))'),'fro');
end
[~, ind] = sort(err,'descend');

% 
% importance = vardist.gamma.*vardist.muW.^2;
% for k=1:size(importance,1)
%     [nada,ind2] = sort(importance(k,:),'descend');
%     importance(k,:) = 0; 
%     importance(k,ind2(1:25)) = 1;
% end
% 
% % mean reconstruction of the training data
% clear predF
% predF = vardist.muPhi*((vardist.gamma.*vardist.muW.*importance)');
% predF = predF*model.scale+ones(n,1)*model.meanY;
% 
% num = zeros(Nh,Nw);
% den = zeros(Nh,Nw);
% for h = 1:B:Nx
%     for w = 1:B:Ny
%         num(h:h+B-1,w:w+B-1) = num(h:h+B-1,w:w+B-1) + reshape(predF(:,(h-1)*Ny+w),B,B);
%         den(h:h+B-1,w:w+B-1) = den(h:h+B-1,w:w+B-1) +1;
%     end
% end
% truncpicest = num./den;
% 
% truncpicest = reshape((truncpicest(:) - mean(truncpicest(:)))*sqrt(var(picorig(:))/var(truncpicest(:)))+mean(picorig(:)),256,256);
% 
% PSNRtruncpicest = 20*log10(255/sqrt(mean(mean((truncpicest-picorig).^2))));
% 
% subplot(2,2,4);image(truncpicest); xlabel(sprintf('PSNR %.2f dB',PSNRtruncpicest)); axis image
colormap((0:255)'/255*[1 1 1])


% Resulting dictionary
rubber = 1;
dict = 255*ones(sqrt(M)*(B+2*rubber));
for i = 1:size(dict,1)/(B+2*rubber)
    for j = 1:size(dict,2)/(B+2*rubber)
        basis = ind((i-1)*size(dict,2)/(B+2*rubber)+j);
        patch = vardist.muPhi(:,basis);
        patch(patch<0.1)=0.1;patch(patch>255)=255;patch = patch/norm(patch);
        dict((i-1)*(B+2*rubber)+rubber+1:(i-1)*(B+2*rubber)+rubber+B, (j-1)*(B+2*rubber)+rubber+1:(j-1)*(B+2*rubber)+rubber+B) = ...
            reshape(patch*20*B^2,B,B);
    end
end

figure
image(dict)
colormap((0:255)'/255*[1 1 1])
axis image
axis off
print -depsc  ./NIPS/figs/houseOU
if train
    save(['house_resultOU_' num2str(noiselevel)])
end
end