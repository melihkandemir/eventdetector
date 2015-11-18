% DEMO  on 10M movielens dataset (ra partition)
clear;
addpath misc_toolbox/;
addpath misc_toolbox/gplm/;


% Fix seeds
randn('seed', 1e5);
rand('seed', 1e5);

% Load 10M movielens data
ddir = 'MOVIELENS_DATA/'; 
% global number of  films
nFilms = 10681;

if ~exist([ddir, '10MraData.mat'])
     % load all data to get the unique films and unique users 
     [users, films, ratings, times] = textread([ddir, 'ratings.dat'], '%n::%n::%n::%n');

     % this also sorts the numbers
     uniqueUsers = unique(users); 
     uniqueFilms = unique(films); 
     InvuniqueFilms = zeros(1,  max(uniqueFilms));
     InvuniqueFilms(uniqueFilms) = 1:size(uniqueFilms,1);

     % Load training Data
     [users, films, ratings, times] = textread([ddir, 'ra.train'], '%n::%n::%n::%n');
     uniqueUsersTr = unique(users); 
     nTrainUsers = size(uniqueUsersTr,1);
     nTrainRatings = size(users,1);
     %Y = spalloc(nTrainUsers, nFilms, nTrainRatings);

     tmp = zeros(1, nFilms);
     Y = sparse([]);
     for n=1:nTrainUsers
         ind = find(users==uniqueUsersTr(n));
         t = tmp;
         t(InvuniqueFilms(films(ind))) = ratings(ind);
         Y = [Y; sparse(t)];
         %Y(n, InvuniqueFilms(films(ind))) = ratings(ind)';
     end
  
     [users_test, films_test, ratings_test, times_test] = textread([ddir, 'ra.test'], '%n::%n::%n::%n');
     uniqueUsersTs = unique(users_test); 
     % Test Data
     nTestUsers = size(uniqueUsersTs,1);
     nTestRatings = size(users_test,1);
     %Ytest = spalloc(nTestUsers, nFilms, nTestRatings);     

     tmp = zeros(1, nFilms);
     Ytest = sparse([]);
     for n=1:nTestUsers
         ind = find(users_test==uniqueUsersTs(n));
         t = tmp;
         t(InvuniqueFilms(films_test(ind))) = ratings_test(ind);
         Ytest = [Ytest; sparse(t)];
         %Ytest(n, InvuniqueFilms(films_test(ind))) = ratings_test(ind);
     end
else
     load([ddir, '10MraData.mat']);
end

indexPresent = Y; 
indexPresent(indexPresent~=0)=1;

% number of latent dimensions
M = 20;
% number of EM iterations
Iters = 50;

tic; 
meanY = sum(Y,2);
l = sum(indexPresent,2); 
meanY = full(meanY./l);
stdY =  sum(Y.*Y,2)./l - meanY.^2;
stdY(stdY<=0) = 1;
stdY = full(sqrt(stdY));
YY = sparse([]);
for i=1:size(Y,2)
    tmp = ((Y(:, i) - meanY)./stdY).*indexPresent(:, i);
    YY = [YY, sparse(tmp)];
    if mod(i,1000)==0
        i
    end
end
Y = YY;
toc;      


for m=1:M
    kernels{m} = {'stadNormal'};
end
% bias/constant kernel
%kernels{M} = {'const'};

noise = 'heterosc'; % anything else will mean ouput-specific (heteroscedastic) noise 
model = varmkgpCreate([], Y, 'Gaussian', kernels, noise, indexPresent);

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


% Training the model with missing values
[model vardist margLogL] = varmkgpMissDataTrain(model, options);

% Reconstruction of the ratings for all users in the training data
% (Compute training error -> just for reference )
tmp = (vardist.gamma.*vardist.muW)';
L2 = 0;
L1 = 0;
round_L1 = 0;
normC = 0; 
predY = sparse([]);
for i = 1:size(Ytest,1)      
  ind = find(Ytest(i, :));
  % compute the mean prediction 
  mu = vardist.muPhi(i,:)*tmp;
  
  % Transform in the original space
  mu(ind) = mu(ind).*(stdY(i)');
  mu(ind) = mu(ind)+meanY(i)'; 
   
  % collect all predictions in a big sparse matrix
  mu(Ytest(i, :)==0) = 0;
  predY = [predY; sparse(mu)];
  diff = Ytest(i, ind) - mu(ind); 
  normC = normC + length(diff);
  L2  = L2  + diff*diff';
  L1 = L1 + sum(abs(diff));
  round_L1 = round_L1 + sum(abs(round(diff)));
end
RMSE_error = sqrt(L2/normC);
NMAE_error = (L1/normC)/1.6;
NMAE_round_error = (round_L1/normC)/1.6;

save demo_MovieLens10Mra_result RMSE_error NMAE_error NMAE_round_error;
