function [ h ] = getHist( block , numwords )
h = hist(block.data(:), numwords);
h = shiftdim(h,-1);
end

