function feature = appFeatExtract_ColorLBP(I,train,camID,numPatch,imEdge)
%feature = appFeatExtract_ColorLBP(I,train,camID,numPatch,imEdge)
%   Color mean + LBP feature for re-id
%   INPUT:  I       - image sequences
%           train   - index of the training sample
%           camID   - camera ID
%           numPatch- number of patches inside the bounding box
%           imEdge  - simantic edge images
%   OUTPUT: feature[NxD]  - features
%   Author: Mengran Gou 
%   Date: 09/16/2016
%   Ref: M. Hirzer et. al. "Relaxed pairwise learned metric for person re-identification"


imsz = [128 64];
step = [4,8];
BBoxsz = [8 16];
LBP_Mapping = getmapping(8,'riu2');

[region_idx, BBox, region_mask] =GenerateGridBBox(imsz, BBoxsz, step);
tmpF = zeros(numel(I),numel(region_idx)*(6+LBP_Mapping.num));

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
disp('Begin ColorLBP extaction...');tic
parfor i = 1:numel(I)
    tmpSeq = I{i};
    tmpfeat = zeros(numel(tmpSeq),numel(region_idx)*(6+LBP_Mapping.num));
    for n = 1:numel(tmpSeq)
        tmpI = tmpSeq{n};
        tmpHSV = zeros(1,numel(region_idx)*3);
        tmpLab = zeros(1,numel(region_idx)*3);
        imHSV = rgb2hsv(uint8(tmpI));
        imLab = rgb2lab(uint8(tmpI));
        for bb = 1:numel(region_idx)
            imH = imHSV(:,:,1);
            imS = imHSV(:,:,2);
            imV = imHSV(:,:,3);
            tmpHSV((bb-1)*3+1) = mean(imH(region_mask(:,bb)&fgzone{camID(1)}));
            tmpHSV((bb-1)*3+2) = mean(imS(region_mask(:,bb)&fgzone{camID(1)}));
            tmpHSV((bb-1)*3+3) = mean(imV(region_mask(:,bb)&fgzone{camID(1)}));
        end
        tmpHSV = (tmpHSV./max(tmpHSV))*40;
        for bb = 1:numel(region_idx)
            imL = imLab(:,:,1);
            imA = imLab(:,:,2);
            imB = imLab(:,:,3);
            tmpLab((bb-1)+1) = mean(imL(region_mask(:,bb)&fgzone{camID(1)}));
            tmpLab((bb-1)+2) = mean(imA(region_mask(:,bb)&fgzone{camID(1)}));
            tmpLab((bb-1)+3) = mean(imB(region_mask(:,bb)&fgzone{camID(1)}));
        end
        tmpLab = (tmpLab./max(tmpLab))*40;
        
        % lbp
        tmpGary = rgb2gray(im2double(tmpI));        
        tmpGary = tmpGary.*reshape(fgzone{camID(i)},imsz);
        tmpLBP = zeros(1,numel(region_idx)*LBP_Mapping.num);
        for bb = 1:size(BBox,1)
            tmpLBP((bb-1)*LBP_Mapping.num+1:bb*LBP_Mapping.num) = ...
                lbp(tmpGary(BBox(bb,2):BBox(bb,4), BBox(bb,1):BBox(bb,3),:),2,LBP_Mapping.samples,LBP_Mapping,'nh');
        end
        tmpColor = [tmpHSV,tmpLab];
        tmpColor(isnan(tmpColor)) = 0;
        tmpColor = normc_safe(tmpColor');
        tmpColor = tmpColor';
        
        tmpLBP(isnan(tmpLBP)) = 0;
        tmpLBP = normc_safe(tmpLBP');
        tmpLBP = tmpLBP'
        
        tmpfeat(n,:) = [tmpColor,tmpLBP];
    end
    tmpF(i,:) = mean(tmpfeat,1);
end
tmpF(isnan(tmpF)) = 0;
feature = tmpF;
disp('Done!');toc