function [X,y,coordslist,frameslist]=prepare_dataset(data,frameList,stackDepth,coords,frames,halfPatchSize)
    X=[];
    y=[];
    
    coordslist=[];
    frameslist=[];
    
    %if matlabpool('size') == 0
    %    matlabpool;
    %end    

    %parfor ff=1:length(frameList)
    for ff=1:length(frameList)

        curFrm=frameList(ff);
        
        coordList=coords(frames==curFrm ,:);
        for cc=1:size(coordList,1)
            
            coordX=coords(cc,1)-halfPatchSize+1:coords(cc,1)+halfPatchSize;
            coordY=coords(cc,2)-halfPatchSize+1:coords(cc,2)+halfPatchSize;
             
            data_now =data(coords(:,1)==coordList(cc,1) & coords(:,2)==coordList(cc,2) & frames<=(curFrm-stackDepth+1) & frames>=curFrm,:);
            data_now =data_now(:)';
           
            data_prev=data(coords(:,1)==coordList(cc,1) & coords(:,2)==coordList(cc,2) & frames<=(curFrm-stackDepth*2+1) & frames>=(curFrm-stackDepth),:);            
            data_prev =data_prev(:)';

            X=[X; data_prev];
            y=[y; data_now];
           coordslist=[coordslist; coordList(cc,:)];        
           frameslist=[frameslist; curFrm];            
        end                       
    end
end