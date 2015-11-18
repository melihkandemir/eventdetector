function [auc,prbep]=draw_pr_curve(vidno,modelno)
    warning off;
    addpath ../../util/
	annots=load(['/home/mkandemi/Documents/data/zeiss/video' num2str(vidno) '/video' num2str(vidno) '_annot.txt']);
    
    if vidno >=3 && vidno <= 5
        annots2=load(['/home/mkandemi/Documents/data/zeiss/video' num2str(vidno) '/video' num2str(vidno) '_annot_mitosis.txt']);        
        annots=[annots; annots2];
    end
    
    load(['/home/mkandemi/Documents/results/zeiss/video' num2str(vidno) '_results_model' num2str(modelno) '_ver2']);
    
    halfPatch=floor(patchSize/2);
    
    [h,w,f]=size(Iheat_ts);
        
    Xctr=halfPatch:patchSize:(h-halfPatch); 
    Yctr=halfPatch:patchSize:(w-halfPatch);  
    
    thrList=unique(Iheat_ts(:));
        
    thrList=linspace(min(thrList),max(thrList),20);
    
    prec=0;
    recall=0;
    
    for ii=1:length(thrList)
        fprintf('Threshold %d / %d\n',ii,length(thrList));
        metrics=compute_event_performance(vidno,Iheat_ts,testFrames,thrList(ii),patchSize);
        prec(ii)=metrics.precision;
        recall(ii)=metrics.recall;
    end
    
    prec(recall==0)=[];
    recall(recall==0)=[];
    
    [rval,ridx]=sort(recall);
    recall=rval;
    prec=prec(ridx);
    
    prec=[1 prec];
    recall=[0 recall];
    auc=trapz(recall,prec);
    
    [~,idx]=min(abs(prec-recall));
    
    prbep=mean([prec(idx) recall(idx)]);
    
    save(['/home/mkandemi/Dropbox/results/zeiss/video' num2str(vidno) '_performance_model' num2str(modelno)],'prec','recall','prbep','auc');    
     
end