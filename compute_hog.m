function norm_hist = compute_hog(block_struct, numBins)

angles = block_struct.data(:,:,1);
magnitudes = block_struct.data(:,:,2);

% Compute the bin size in radians. 180 degress = pi.
binSize = pi / numBins;

% Make the angles unsigned by adding pi (180 degrees) to all negative angles.
angles(angles < 0) = angles(angles < 0) + pi;

%compute histograms in 2by2grid
idx_image = im2col(reshape(1:50*50, 50, 50), [25 25], 'distinct');
angle_cells = im2col(angles, [25 25], 'distinct');
magn_cells = im2col(magnitudes, [25 25], 'distinct');

%recover indexes of pixels in each angle-bin
[~, Y] = histc(angles(idx_image),0:binSize:7*binSize);
hist = zeros(4, numBins);
for i = 1:size(Y,2)
    for j = 1:numBins        
        %compute histogram as the angle values weighted with the magintude
        hist(i,j) = hist(i,j) + angle_cells(Y(:,i)+1 == j,i)' * magn_cells(Y(:,i)+1 == j,i);
    end   
end

%normalize histograms within each block
norm_hist = hist./repmat(sqrt(sum((hist+eps).^2,2)), 1, size(hist,2));
norm_hist = norm_hist(:)';
end

