function feature = appFeatExtract_HistLBP(I,train,camID,numPatch,imEdge)
%feature = appFeatExtract_HistLBP(I,train,camID,numPatch,imEdge)
%   Color histgram + LBP featuer for person re-id
%   INPUT:  I       - image sequences
%           train   - index of the training sample
%           camID   - camera ID
%           numPatch- number of patches inside the bounding box
%           imEdge  - simantic edge images
%   OUTPUT: feature[NxD]  - features
%   Author: Mengran Gou 
%   Date: 09/16/2016
%   Ref: F. Xiong et. al. "Person re-identification using kernel-based metric learning methods"

step =flipud([8 10; 16 16; 21 64; 64 32; 128 64]); % set the moving step size of the region.
BBoxsz =flipud([16 21; 32 32; 22 64; 64 32; 128 64]); % set the region size.
numP = [1 4 6 14 75];
imsz = [128 64];
kk = find(numP == numPatch);
[region_idx, BBox, region_mask] =GenerateGridBBox(imsz, BBoxsz(kk,:), step(kk,:));

n8LBP_Mapping = getmapping(8,'u2');
n16LBP_Mapping = getmapping(16,'u2');
num_colorChn = 8;
num_bin = 16;

uCam = unique(camID); 

% apply edge mask if provided
if ~isempty(imEdge)
    for c = 1:numel(uCam)
        idxC = find(camID==uCam(c));
        idxC_train = intersect(train,idxC);
        % filter out background based on edge mask 
        meanEdge = cell2mat(imEdge(idxC_train));
        meanEdge = reshape(meanEdge,128,64,[]);
        meanEdge = mean(meanEdge,3);

        meanEdge(meanEdge<mean(meanEdge(:)))=0;
        meanEdge(meanEdge>0) = 1;
        se = strel('disk',5,4);
        tmpEdge = imdilate(meanEdge,se);
        tmpEdge([1 128],:) = 0;
        tmpEdge(:,[1 64]) = 0;
        tmpEdge = imfill(tmpEdge,'holes');
        meanEdge = tmpEdge;
        % meanEdge = imerode(tmpEdge,se);
        fgzone{uCam(c)} = meanEdge(:)==1;
    end
else
    uCam = unique(camID); % one mask per camera
    for c = 1:numel(uCam)
        fgzone{uCam(c)} = logical(ones(prod(imsz),1));
    end
end

tmpF = zeros(numel(I),numel(region_idx)*(num_bin*num_colorChn+n8LBP_Mapping.num+n16LBP_Mapping.num));
dim_color = num_bin*num_colorChn;
dim_LBP = n8LBP_Mapping.num+n16LBP_Mapping.num;

bin = repmat(1e-5:1/num_bin:1,6,1);
bin = [bin; [16:((235-15)/num_bin):235] ./255];
bin = [bin; [16:((240-15)/num_bin):240] ./255];

disp('Begin HistLBP extaction...');tic
parfor i = 1:numel(I)
    tmpSeq = I{i};
    tmpF_color = zeros(numel(tmpSeq),num_bin*num_colorChn,numel(region_idx));
    tmpF_lbp = zeros(numel(tmpSeq),n8LBP_Mapping.num+n16LBP_Mapping.num,numel(region_idx));
    for f = 1:numel(tmpSeq)
        tmpRGB = im2double(tmpSeq{f});
        tmpHSV = rgb2hsv(tmpRGB);
        tmpYUV = rgb2ycbcr(tmpRGB);
        tmpGray = rgb2gray(tmpRGB);
        tmpGray = tmpGray.*reshape(fgzone{camID(i)},imsz);
        tmpChn = cat(3,tmpRGB, tmpHSV, tmpYUV(:,:,1:2));
        for c = 1:size(tmpChn,3)
            tmp_ch = tmpChn(:,:,c);
            for bb = 1:size(region_mask,2)
                tmpHist = hist(tmp_ch(region_mask(:,bb) & fgzone{camID(i)}),bin(c,:));
                tmpF_color(f,(c-1)*num_bin+1:c*num_bin,bb) = tmpHist./sum(tmpHist);
            end
        end
        
        for bb = 1:numel(region_idx)
            tmpF_lbp(f,1:n8LBP_Mapping.num,bb) = lbp(tmpGray(BBox(bb,2):BBox(bb,4), ...
                BBox(bb,1):BBox(bb,3),:),1,n8LBP_Mapping.samples,n8LBP_Mapping,'nh')';
            tmpF_lbp(f,n8LBP_Mapping.num+1:end,bb) = lbp(tmpGray(BBox(bb,2):BBox(bb,4), ...
                BBox(bb,1):BBox(bb,3),:),2,n16LBP_Mapping.samples,n16LBP_Mapping,'nh')';
        end        
    end
    tmpF_color = reshape(tmpF_color,size(tmpF_color,1),[]);
    tmpF_color(isnan(tmpF_color)) = 0;
    tmp_feat = normc_safe(tmpF_color');
    tmpF_color = tmp_feat';
    
    tmpF_lbp = reshape(tmpF_lbp,size(tmpF_lbp,1),[]);
    tmpF_lbp(isnan(tmpF_lbp)) = 0;
    tmp_feat = normc_safe(tmpF_lbp');
    tmpF_lbp = tmp_feat';
    
    tmpF(i,:) = [mean(tmpF_color,1) mean(tmpF_lbp,1)];
end
feature = tmpF;
disp('Done!');toc
