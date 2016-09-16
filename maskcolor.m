function ChnFeat = maskcolor(I,imsz)
% coordinate + color
num_frame = numel(I);
numChn = 5;
ChnFeat = zeros(numChn,imsz(1)*imsz(2),num_frame, 'single');
for f = 1:num_frame
    tmp_im = imresize(I{f},imsz);
    tmp_data = zeros(imsz(1),imsz(2),numChn);
    % get the channel feature
    tmp_data(:,:,1) = repmat([1:imsz(1)]',1,imsz(2));
    tmp_data(:,:,1) = tmp_data(:,:,1)./imsz(1);
    tmp_data(:,:,2) = repmat([1:imsz(2)],imsz(1),1);
    tmp_data(:,:,2) = tmp_data(:,:,2)./imsz(2);
    tmp_hsv = rgb2hsv(tmp_im);
    tmp_data(:,:,3:5) = tmp_hsv;
    
    tmp_data = permute(tmp_data, [3 1 2]);
    tmp_data = reshape(tmp_data,size(tmp_data,1),[]);
    ChnFeat(:,:,f) = tmp_data;
end