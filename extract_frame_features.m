function features=extract_frame_features(I,Iprevious,frameno,patchSize)

    addpath ./vlfeat/toolbox/
    run('vl_setup');
    warning off;
    
    if length(size(I))>2
        I=rgb2gray(I);
        Iprevious=rgb2gray(Iprevious);
    end
    
    % config
    bins=1:10:255;    
    halfPatch=floor(patchSize/2);    
   
    [h,w,~]=size(I);    
            
    XctrList=halfPatch:patchSize:(h-halfPatch);
    YctrList=halfPatch:patchSize:(w-halfPatch);        
    [II, JJ] = meshgrid(1:length(XctrList), 1:length(YctrList));
    all_coords = [XctrList(II(:))' YctrList(JJ(:))'];
    
    numBinsAngleHog = 8;
    data_mat=[];
    magnif = 3;
    lbp_bin_size = 10;
    binSize = patchSize;
    lbp_dims_per_patch = floor(patchSize/lbp_bin_size);
    bins_bhist  = 1:10:255;
    flbp_all_im = cell(1, 250);
    sift_frames = cell(1, 250);
    d = cell(1, 250);    
    fun = @(block_struct) getHist(block_struct, bins_bhist);    
    fun_hog = @(block_struct) compute_hog(block_struct, numBinsAngleHog);
    zero_col = zeros(length(XctrList)*length(YctrList),1);
    data = cell(1, 250);
    hog_dims = 2*2*numBinsAngleHog;
    
    %Create the operators for computing image derivative at every pixel.
    hx = [-1,0,1];
    hy = hx';

    
        croppedI = I(1:XctrList(end)+halfPatch,1:YctrList(end)+halfPatch);        
        croppedIprev = Iprevious(1:XctrList(end)+halfPatch,1:YctrList(end)+halfPatch);        
        
        %SIFT
        Is = vl_imsmooth(double(I), sqrt((binSize/magnif)^2 - .25)) ;
        [~, d] = vl_dsift(single(Is), 'size', patchSize/3, 'Step', patchSize, 'Fast') ;                
        
        IsPrev = vl_imsmooth(double(abs(I - Iprevious)), sqrt((binSize/magnif)^2 - .25)) ;
        [~, d_diff] = vl_dsift(single(IsPrev), 'size', patchSize/3, 'Step', patchSize, 'Fast') ;                
                        
        %IsPrev = vl_imsmooth(double(abs(Iprevious)), sqrt((binSize/magnif)^2 - .25)) ;
        %[~, d_prev] = vl_dsift(single(IsPrev), 'size', patchSize/3, 'Step', patchSize, 'Fast') ;                
        %d_diff = d - d_prev;
        
        %LBP
        flbp_all_im = vl_lbp(single(croppedI),lbp_bin_size);
        all_lbs_cell = mat2cell(flbp_all_im, ones(1, length(XctrList))*lbp_dims_per_patch, ones(1, length(YctrList))*lbp_dims_per_patch, size(flbp_all_im,3));
        all_lbs_cell = all_lbs_cell';
        flbp = cell2mat(cellfun(@(x)(squeeze(mean(mean(x,1),2))'), all_lbs_cell(:), 'UniformOutput', false));
        
        %Brightness histograms 
        hist_p = reshape(blockproc(double(croppedI'), [patchSize patchSize], fun), [], length(bins_bhist));
        hist_diff_p = reshape(blockproc(double(abs(croppedI' - croppedIprev')), [patchSize patchSize], fun), [], length(bins_bhist));
        
        %% Small single-scale HOG features
        % Compute the derivative in the x and y direction for every pixel.
        dx = imfilter(double(croppedI), hx);
        dy = imfilter(double(croppedI), hy);

        % Convert the gradient vectors to polar coordinates (angle and magnitude).
        angles = atan2(dy, dx);
        magnit = ((dy.^2) + (dx.^2)).^.5;    

        %for each patch compute a 2 by 2 grid and compute histograms of angles
        %within
        Hog_features = reshape(blockproc(double(cat(3, angles, magnit)), [patchSize patchSize], fun_hog), [], hog_dims);
        %%
        
        %0 0 0 0 Xctr Yctr ff 0 0 hist_patch diff_hist_patch flbp fsift        
        features= [zero_col zero_col zero_col zero_col all_coords(:,1) all_coords(:,2) (1 - zero_col)*frameno zero_col zero_col hist_p hist_diff_p flbp double(d)' double(d_diff)', Hog_features];
               

    
end
