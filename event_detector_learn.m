function modelstruct=event_detector_learn(TrainData,modelno,trainFrames,patchSize,ImageSize)
    
    stackDepth=1; % based on how many frames the autoregression will be performed    
    Dred=10;    
    
    data=TrainData(:,10:end);    
    data=double(data);
    data(isnan(data)==1)=0;         
    
    coords=TrainData(:,5:6);
    frames=TrainData(:,7);
    
    % constants
    halfPatchSize=floor(patchSize/2);    
       
    
    if modelno>4 % dictionary learning, then no need for dimensionality reduction
        traindatamean=0; % not used
        traindatastd=0; % not used
    else
        traindatamean=mean(data);
        traindatastd=std(data);
        traindatastd(traindatastd==0)=1;        

        data=(data-repmat(traindatamean,size(data,1),1)) ./ repmat(traindatastd,size(data,1),1);


        [W{1},data1,~]=pca(data(:,1:26)); % HIST 26
        data1=data1(:,1:Dred);

        [W{2},data2,~]=pca(data(:,27:52));%DIFF HIST 26
        data2=data2(:,1:Dred);    

        [W{3},data3,~]=pca(data(:,53:110));%LBP 58    
        data3=data3(:,1:Dred);

        [W{4},data4,~]=pca(data(:,111:238)); %SIFT 128    
        data4=data4(:,1:Dred);

        [W{5},data5,~]=pca(data(:,239:366)); %DIFF_SIFT    
        data5=data5(:,1:10);

        [W{6},data6,~]=pca(data(:,367:398)); %HOG 32    
        data6=data6(:,1:10);

        data=[data2 data3 data4 data5 data6];    
    end
    
    % prepare the training set
    [Xtr,ytr,coordstr,framestr]=prepare_dataset(data,trainFrames,stackDepth,coords,frames,halfPatchSize);
        
    [Ntr,D]=size(Xtr);
    
    % train the model
    if modelno==1
        addpath ./varsparse_mtmkl/
        addpath ./varsparse_mtmkl/misc_toolbox/;
        addpath ./varsparse_mtmkl/misc_toolbox/gplm/;     
        kernels{1}='covSEiso_fp_view1';
        kernels{2}='covSEiso_fp_view2';
        kernels{3}='covSEiso_fp_view3';
        kernels{4}='covSEiso_fp_view4';
        kernels{5}='covSEiso_fp_view5';
        %kernels{6}='covSEiso_fp_view6';        
        
        noise = 'homosc';
          
        model = varmkgpCreate(Xtr, ytr,'Gaussian', kernels, noise);     

        options=loadDefaultOptions();

        % do few steps with sifex sigma2 for initialization 
       tic;
       [mdl vardist margLogL] = varmkgpMissDataTrain(model, options);
       duration=toc;
       
       model.varParams=mdl;
       model.vardist=vardist;
       model.margLogL=margLogL;
             
    elseif modelno==2 % Plain GP
        addpath ./gpml/
        addpath ./gpml/cov/
        addpath ./gpml/inf/
        addpath ./gpml/likframefeatures
        addpath ./gpml/util
        addpath ./gpml/mean        
        
        kernels='';
          
        meanfunc =@meanConst; hyp.mean = 0;
        covfunc = @covSEiso; hyp.cov = log([1; 1]);        
        hyp.lik=1;

        likfunc = @likGauss;

        % do few steps with sifex sigma2 for initialization 
        tic;
        for dd=1:D
            
            model{dd}.hyp = minimize(hyp, @gp, -20, @infExact, meanfunc, covfunc, likfunc, Xtr, ytr(:,dd));
            model{dd}.meanfunc=meanfunc;
            model{dd}.covfunc=covfunc;
            model{dd}.likfunc=likfunc;                               
          
        end        
        duration=toc;        
    elseif modelno==3 % One Class SVM
        addpath ./libsvm/
        rmpath ./vlfeat/toolbox/noprefix/
        kernels='';        
                
        tic;
        model=svmtrain(ones(Ntr,1),Xtr,'-s 2 -t 2  -n 0.5');                 
        duration=toc;        
    elseif modelno==4 % linear regression
        kernels='';  
        tic;
        for dd=1:D        
           model{dd}.w=regress(ytr(:,dd),Xtr);
        end
        duration=toc;
    else
        addpath ./nmf_bpas/        
        kernels='';
        numDictElements=20;
        tic;
        [W,H,iter,HIS]=nmf(ytr,numDictElements); 
        
        Wsum=sum(W,2);
        
        %W = W ./ repmat(sum(W,2),1,size(W,2));

        %for rr=1:size(W,1)
        %   Etrain(rr)=calculate_entropy(W(rr,:));
        %end
        
        model.W=W;
        model.H=H;
       % model.Etrain=Etrain;
        model.MaxTrainActivation=mean(Wsum(:))+3*std(Wsum(:));
        model.numDictElements=numDictElements;

        duration=toc;
    end        
     
    modelstruct.model=model;
    modelstruct.Xtr=Xtr;
    modelstruct.ytr=ytr;
    modelstruct.patchSize=patchSize;
    modelstruct.stackDepth=stackDepth;
    modelstruct.W=W;
    modelstruct.traindatamean=traindatamean;
    modelstruct.traindatastd=traindatastd;
    modelstruct.Dred=Dred;
    modelstruct.ImageSize=ImageSize;
    modelstruct.modelno=modelno;
    
    % Calculate training error to be used as a reference
    if modelno <=4
        trainErrors=[];
        for ff=1:length(trainFrames)
            Iheat_tr=event_detector_predict(modelstruct,TrainData,trainFrames(ff));
            trainErrors=[trainErrors; Iheat_tr(:)];
        end
        modelstruct.trainErrorMean=mean(trainErrors);
        modelstruct.trainErrorStd=std(trainErrors);
    else
        maxActivations=zeros(ImageSize(1:2));
        for ff=1:length(trainFrames)
            Iheat=event_detector_predict(modelstruct,TrainData,trainFrames(ff));
            maxActivations=max(maxActivations,Iheat);
        end  
        modelstruct.model.maxTrainActivations=maxActivations;
    end
end

function options=loadDefaultOptions()
    load defoptions;
    options(1) = 0; % display lower bound during running...
    options(2) = 0; % learn kernel hyperparameters (0 for not learning)... 
    options(3) = 1; % learn sigma2W hyperprameter (0 for not learning)...
    options(4) = 1; % learn likelihood noise parameters sigma2 (0 for not learning)...
    options(5) = 1; % learn pi sparse mixing coefficient (0 for not learning)...
    options(10) = 1; % use sparsity or not (if not pi is set to 1, is not learned)..
    options(11) = 10; % number of variational EM iterations; % FOR FINAL RUNS, SET THIS TO 100!!!!!
    
end
