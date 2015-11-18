% Demo for inpainting
clear all;
close all;
addpath misc_toolbox/;
addpath misc_toolbox/gplm/;


train = 1; % Set to 0 if you want to load and display results

% Fix seeds
randn('seed', 1e5);
rand('seed', 1e5);

% number of GP latent GP functions 
M = 100;
Iters = 60; 

% load data
picorig = double(imread('castle.png'));
[Nh, Nw, ncolors] = size(picorig);
pic = picorig;
IndexObserved = rand(length(pic(:)),1)>0.8;

pic(IndexObserved==0) = 0;
PSNRpic = 20*log10(255/sqrt(mean((pic(:)-picorig(:)).^2)))
pic(IndexObserved==0) = nan;

image(uint8(pic)); axis image

B=8; % block size is BxB
n = B*B*ncolors;

% Cut in overlapping blocks
Nx = Nh-B+1;
Ny = Nw-B+1;
Y = zeros(n, Nx*Ny);
for h = 1:Nx
    for w = 1:Ny
        block = pic(h:h+B-1,w:w+B-1,:);
        Y(:,(h-1)*Ny+w) = block(:);
        
    end
end

% keep the substracted mean and scale for prediction purposes
meanY=zeros(1,size(Y,2));
for q = 1:size(Y,2)
    meanY(q) = mean(Y(~isnan(Y(:,q)),q));
end
scale = sqrt(var(Y(~isnan(Y(:)))));
Y = (Y - repmat(meanY,size(Y,1),1))/scale;

    
kernels={};
for m=1:M 
    kernels{m} = {'stadNormal'};
end

noise = 'homosc'; % anything else will mean ouput-specific (heteroscedastic) noise 

% Create the model structure 
model = varmkgpCreate([], Y, 'Gaussian', kernels, noise);
clear Y
model.meanY = meanY;
model.scale = scale; 

model.Likelihood.sigma2=600/model.scale^2; % better initial guess

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

if train == 1
    [model vardist margLogL] = varmkgpMissDataTrain(model, options);
    save inpainting_result
else
    load inpainting_result
end

% mean reconstruction of the training data
predF = vardist.muPhi*((vardist.gamma.*vardist.muW)');
predF = predF*model.scale+ones(n,1)*model.meanY;

num = zeros(Nh,Nw,ncolors);
den = zeros(Nh,Nw,ncolors);
for h = 1:Nx
    for w = 1:Ny
        num(h:h+B-1,w:w+B-1,:) = num(h:h+B-1,w:w+B-1,:) + reshape(predF(:,(h-1)*Ny+w),B,B,ncolors);
        den(h:h+B-1,w:w+B-1,:) = den(h:h+B-1,w:w+B-1,:) +1;
    end
end
picest = num./den;

PSNRpicest = 20*log10(255/sqrt(mean((picest(:)-picorig(:)).^2)))

close all
subplot(1,3,1);image(uint8(picorig));axis image
subplot(1,3,2);image(uint8(pic));xlabel(sprintf('PSNR %.2f dB',PSNRpic)); axis image
subplot(1,3,3);image(uint8(picest)); xlabel(sprintf('PSNR %.2f dB',PSNRpicest)); axis image

% Importance order
[~, ind] = sort(mean(vardist.gamma.*vardist.muW.^2),'descend');

% Resulting dictionary
rubber = 1;
dict = zeros(sqrt(M)*(B+2*rubber),sqrt(M)*(B+2*rubber),ncolors);
for i = 1:size(dict,1)/(B+2*rubber)
    for j = 1:size(dict,2)/(B+2*rubber)
        basis = ind((i-1)*size(dict,2)/(B+2*rubber)+j);
        dict((i-1)*(B+2*rubber)+rubber+1:(i-1)*(B+2*rubber)+rubber+B, (j-1)*(B+2*rubber)+rubber+1:(j-1)*(B+2*rubber)+rubber+B,:) = ...
            reshape(vardist.muPhi(:,basis)*model.scale,B,B, ncolors);
    end
end

figure
image(uint8(dict))
axis image