function ChnFeat = maskT(I,imsz)
% temporal LDFV
num_frame = numel(I);
numChn = 24;
ChnFeat = zeros(numChn,imsz(1)*imsz(2),num_frame-2, 'single');
for f = 1:num_frame-2
    tmp_im = imresize(I{f},imsz);
    tmp_im1 = imresize(I{f+1},imsz);
    tmp_im2 = imresize(I{f+2},imsz);
    tmp_data = zeros(imsz(1),imsz(2),numChn);
    % get the channel feature
    tmp_data(:,:,1) = repmat([1:imsz(1)]',1,imsz(2));
    tmp_data(:,:,1) = tmp_data(:,:,1)./imsz(1);
    tmp_data(:,:,2) = repmat([1:imsz(2)],imsz(1),1);
    tmp_data(:,:,2) = tmp_data(:,:,2)./imsz(2);
    tmp_data(:,:,3) = f/(num_frame-2); % t
    tmp_hsv = rgb2hsv(tmp_im);
    tmp_hsv1 = rgb2hsv(tmp_im1);
    tmp_hsv2 = rgb2hsv(tmp_im2);
    tmp_data(:,:,4:6) = tmp_hsv;
    for c = 1:3
        [tmpX,tmpY] = imgradientxy(reshape(tmp_hsv(:,:,c),imsz(1),imsz(2)));
        tmpT = tmp_hsv(:,:,c)-tmp_hsv1(:,:,c);
        tmpTT = tmp_hsv(:,:,c)-tmp_hsv1(:,:,c)*2+tmp_hsv2(:,:,c);
        [tmpXX,~] = imgradientxy(tmpX);
        [~,tmpYY] = imgradientxy(tmpY);
        tmp_data(:,:,(c-1)*6+7:c*6+6) = cat(3,tmpX,tmpY,tmpT,tmpXX,tmpYY,tmpTT);
    end
    tmp_data = permute(tmp_data, [3 1 2]);
    tmp_data = reshape(tmp_data,size(tmp_data,1),[]);
    ChnFeat(:,:,f) = tmp_data;
end