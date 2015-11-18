function Iheat_ts=event_detector_predict(modelstruct,framefeatures,frameNo)

    patchSize=modelstruct.patchSize;
    stackDepth=modelstruct.stackDepth;
    model=modelstruct.model;
    Xtr=modelstruct.Xtr;
    ytr=modelstruct.ytr;
    imgx=modelstruct.ImageSize(1);
    imgy=modelstruct.ImageSize(2);
    modelno=modelstruct.modelno;
    Dred=modelstruct.Dred;
    
    halfPatchSize=floor(patchSize/2);   
    
    data=framefeatures(:,10:end);    
    data=double(data);
    data(isnan(data)==1)=0;         
    
    coords=framefeatures(:,5:6);
    frames=framefeatures(:,7);    
    
    % Normalize and pca    
    if modelstruct.modelno<=4 % not dictionary learning
        data=(data-repmat(modelstruct.traindatamean,size(data,1),1)) ./ repmat(modelstruct.traindatastd,size(data,1),1);        

        data1=data(:,1:26)*modelstruct.W{1};
        data1=data1(:,1:Dred);

        data2=data(:,27:52)*modelstruct.W{2};
        data2=data2(:,1:Dred);        

        data3=data(:,53:110)*modelstruct.W{3};
        data3=data3(:,1:Dred);

        data4=data(:,111:238)*modelstruct.W{4};
        data4=data4(:,1:Dred);

        data5=data(:,239:366)*modelstruct.W{5};
        data5=data5(:,1:Dred);

        data6=data(:,367:398)*modelstruct.W{6};
        data6=data6(:,1:Dred);

        data=[data2 data3 data4 data5 data6];     
    end
    
    % prepare the test set
    [Xts,yts,coordsts,framests]=prepare_dataset(data,frameNo,stackDepth,coords,frames,halfPatchSize);
    
    [Nts,D]=size(Xts);
    
    if modelno==1
        ypred = varmkgpPredict(model.varParams, model.vardist, Xts);  
        
        errts=sqrt(mean((ypred-yts).^2,2)); 
    elseif modelno==2
        
        for dd=1:D
            [ypred_dd s2] = gp(model{dd}.hyp, @infExact, model{dd}.meanfunc, model{dd}.covfunc, model{dd}.likfunc, Xtr, ytr(:,dd), Xts);            
            ypred(:,dd) = ypred_dd;      
        end    
        
        errts=sqrt(mean((ypred-yts).^2,2));         
    elseif modelno==3
        [ypred, ~, probs] = svmpredict(ones(Nts,1), Xts, model); 
        errts=-probs;        
    elseif modelno==4
        for dd=1:D
            ypred(:,dd)= Xts*model{dd}.w;
        end
        
        errts=sqrt(mean((ypred-yts).^2,2));
    else % dictionary learning
      for rr=1:Nts
            ypred(rr,:)=lsqnonneg(model.H',yts(rr,:)');                          
            errts(rr)=sum(ypred(rr,:));
      end         
    end
      
    Iheat_ts(imgx,imgy)=0;            
    
    for ss=1:Nts
       xcoords=max(coordsts(ss,1)-halfPatchSize+1,1) : min(coordsts(ss,1)+halfPatchSize,imgx);
       ycoords=max(coordsts(ss,2)-halfPatchSize+1,1) : min(coordsts(ss,2)+halfPatchSize,imgy);   

     
       Iheat_ts(xcoords,ycoords)=errts(ss);              
    end     

end
