function ChnFeat = maskedge(I,imsz)
% coordinates + gradient
num_frame = numel(I);
numChn = 14;
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
    for c = 1:3
        [tmpX,tmpY] = imgradientxy(reshape(tmp_hsv(:,:,c),imsz(1),imsz(2)));
        [tmpXX,~] = imgradientxy(tmpX);
        [~,tmpYY] = imgradientxy(tmpY);
        tmp_data(:,:,(c-1)*4+3:c*4+2) = cat(3,tmpX,tmpY,tmpXX,tmpYY);
    end
    tmp_data = permute(tmp_data, [3 1 2]);
    tmp_data = reshape(tmp_data,size(tmp_data,1),[]);
    ChnFeat(:,:,f) = tmp_data;
end