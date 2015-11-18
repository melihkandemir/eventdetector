function metrics=compute_event_performance(vidno,Iheat,testFrames,thr,patchSize)
    addpath ../../util/
	annots=load(['/home/mkandemi/Documents/data/zeiss/video' num2str(vidno) '/video' num2str(vidno) '_annot.txt']);
    
     if vidno >=3 && vidno <= 5
        annots2=load(['/home/mkandemi/Documents/data/zeiss/video' num2str(vidno) '/video' num2str(vidno) '_annot_mitosis.txt']);
       
        annots=[annots; annots2];    
     end
     Nannot=size(annots,1);
    
    
     halfPatch=floor(patchSize/2);
    
    [h,w,f]=size(Iheat);
    
    
    Xctr=halfPatch:patchSize:(h-halfPatch); 
    Yctr=halfPatch:patchSize:(w-halfPatch);   
    
    cnt=0;
    
    
    for tt=1:length(testFrames)
        Ipred=predict_event(Iheat(:,:,tt),thr);
                
        for ii=1:length(Xctr)
            for jj=1:length(Yctr)
                if Ipred(Xctr(ii),Yctr(jj))>0
                                       
                    if cnt>0 
                        preds_last5=preds(preds(:,1)>=testFrames(tt)-5 & preds(:,2)==Xctr(ii) & preds(:,3)==Yctr(jj),:);
                    else
                        preds_last5=[];
                    end
                    
                    if isempty(preds_last5)

                        cnt=cnt+1;                        
                        preds(cnt,:)= [testFrames(tt) Xctr(ii) Yctr(jj) Iheat(Xctr(ii),Yctr(jj),tt)];
                    end
                    
                end
            end
        end                        
    end
    
    
    if cnt==0
        metrics.precision=0;
        metrics.recall=0;
        return;
    end
    
    [Npred,Dpred]=size(preds);        
    predVisited=zeros(Npred,1);
    
    ypred=0;
    ytrue=0;
    fval=0;
    cnt=0;
    
    for ii=1:Nannot
        xx=annots(ii,3);
        yy=annots(ii,2);
        ff=annots(ii,1);
        
        if ismember(ff,testFrames)==0
            continue;
        end
        
        predInCorrectRegion=abs(preds(:,2)-xx) <= 30 & abs(preds(:,3)-yy) <= 30 & abs(ff-preds(:,1)) <= 3;
        
        correctPredList=find(predInCorrectRegion==1);
        
        if isempty(correctPredList)
            cnt=cnt+1;
            ypred(cnt)=0;
            ytrue(cnt)=1;
            frameIdx=find(testFrames==ff);
            fval(cnt)=Iheat(uint8(xx),uint8(yy),frameIdx);
        else
            for jj=1:length(correctPredList)
                cnt=cnt+1;
                ypred(cnt)=1;
                fval(cnt)=preds(correctPredList(jj),4);

                predVisited(correctPredList(jj))=1;

                if jj==1
                    ytrue(cnt)=1;
                else
                    ytrue(cnt)=0;
                end

            end 
        end
        
    end
    
    for ii=1:Npred
        if predVisited(ii)==0
            cnt=cnt+1;
            ypred(cnt)=1;
            ytrue(cnt)=0;
            fval(cnt)=preds(ii,4);
        end
    end
    
    if  isempty(ytrue)==0 && isempty(ypred)==0 && isempty(fval)==0
        metrics=compute_performance(ytrue',ypred',fval');      
    else
        metrics.precision=0;
        metrics.recall=0;
        metrics.f1=0;
    end

end
