function features=extract_raw_frame_features(I,Iprevious,frameno,patchsize)

    Idiff=abs(I-Iprevious);
    
    halfPatch=floor(patchsize/2);    
   
    [h,w,~]=size(I);    
            
    xcoord=halfPatch:patchsize:(h-halfPatch);
    ycoord=halfPatch:patchsize:(w-halfPatch);        
    [II, JJ] = meshgrid(1:length(xcoord), 1:length(ycoord));
    all_coords = [xcoord(II(:))' ycoord(JJ(:))'];
    
    zero_col = zeros(length(xcoord)*length(ycoord),1);    
       
    rawdata=[];
    for ii=1:length(xcoord)
       for jj=1:length(ycoord)
              Ipatch=double(I(xcoord(ii)-halfPatch+1:xcoord(ii)+halfPatch,ycoord(jj)-halfPatch+1:ycoord(jj)+halfPatch));
              Ipatch_diff=double(Idiff(xcoord(ii)-halfPatch+1:xcoord(ii)+halfPatch,ycoord(jj)-halfPatch+1:ycoord(jj)+halfPatch));
              rawdata=[rawdata; Ipatch_diff(:)'];
              %rawdata=[rawdata; Ipatch(:)' Ipatch_diff(:)'];
        end
    end    
    
    rawdata=rawdata/255;

    features= [zero_col zero_col zero_col zero_col all_coords(:,1) all_coords(:,2) (1 - zero_col)*frameno zero_col zero_col rawdata];
end
