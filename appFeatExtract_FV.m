function [ F ] = appFeatExtract_FV(I, train, camID, numPatch, hf_chn, imEdge)
%[ F ] = appFeatExtract_FV(I, train, camID, numPatch, hf_chn, imEdge)
%   Fisher Vector encoded appearance feature extraction
%   INPUT:  I       - image sequences
%           train   - index of the training sample
%           camID   - camera ID
%           numPatch- number of patches inside the bounding box
%           hf_chn  - raw feature function
%           imEdge  - simantic edge images
%   OUTPUT: F[NxD]  - features
%   Author: Mengran Gou 
%   Date: 09/16/2016
%   Ref: B.Ma, et. al. "Local descriptors encoded by fisher vectors for person re-identification"


try
    vl_version
catch
    error('Cannot find vl_feat!')
end

step =flipud([8 10; 16 16; 21 64; 64 32; 128 64]); % set the moving step size of the region.
BBoxsz =flipud([16 21; 32 32; 22 64; 64 32; 128 64]); % set the region size.
numP = [1 4 6 14 75];
imsz = [128 64];
numbin = 16;
numChn = 17;%14+2;
kk = find(numP == numPatch);

uCam = unique(camID); % one mask per camera
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

[~, BBox, region_mask] = GenerateGridBBox(imsz, BBoxsz(kk,:), step(kk,:));

chnFeat = cell(1,numel(I));
fprintf('Begin to extract channel feature... \n');tic
for i = 1:numel(I) % per person   
    if mod(i,round(numel(I)/10))==0
            fprintf('.');
    end
    tmpI = I{i};       
    if iscell(tmpI)
        num_frame = numel(tmpI);
    else 
        num_frame = 1;
        tmpI = {tmpI};
    end
    tmpChnFeat = hf_chn(tmpI,imsz);
    chnFeat{i} = tmpChnFeat;
end
numChn = size(chnFeat{1},1);
fprintf('Done!\n');toc
fprintf('Begin to build GMM model...\n');tic
% GMM encoding
idx = 1:numel(I);
idx_train = ismember(idx,train);
for s = 1:numPatch   
    ChnFeat_patch = [];
    for c = 1:numel(unique(camID))
        tmpChnFeat = cellfun(@(x) x(:,logical(region_mask(:,s) & fgzone{uCam(c)}),:), chnFeat(idx_train & camID==uCam(c)),'UniformOutput',0);
        tmpChnFeat = cellfun(@(x) reshape(x,numChn,[]), tmpChnFeat,'UniformOutput',0);
        tmpChnFeat = cell2mat(tmpChnFeat);
        % subsample
        MAXsample = 100000;
        idxsub = randsample(size(tmpChnFeat,2),min(MAXsample,size(tmpChnFeat,2))); 
        tmpChnFeat = tmpChnFeat(:,idxsub);
        ChnFeat_patch = [ChnFeat_patch tmpChnFeat];
    end
    [means{s}, covariances{s}, priors{s}] = vl_gmm(ChnFeat_patch,numbin,'NumRepetitions',1);    
end
fprintf('Done!\n');toc
fprintf('Begin to extract fisher vectors...\n');tic
% FV encode
F = zeros(numel(chnFeat),numChn*2*numbin*numPatch,'single');
for i = 1:numel(chnFeat)
    if mod(i,round(numel(I)/10))==0
            fprintf('.');
    end
    tmpChnFeat = chnFeat{i};
    tmpF_perP = zeros(numChn*2*numbin*numPatch,size(tmpChnFeat,3),'single');
    for f = 1:size(tmpChnFeat,3)
        tmpFrame = tmpChnFeat(:,:,f);
        tmpF_perf = zeros(numChn*2*numbin,numPatch,'single');
        for s = 1:numPatch        
            tmpFrame_s = tmpFrame(:,logical(region_mask(:,s)) & fgzone{camID(i)});
            tmpF_perf(:,s) = vl_fisher(tmpFrame_s,means{s}, covariances{s}, priors{s},'Normalized','SquareRoot');
        end
        tmpF_perP(:,f) = tmpF_perf(:);
    end
    % naive mean along the temporal dimension
    tmpF_mean = mean(tmpF_perP,2);
    tmpF_mean(isnan(tmpF_mean)) = 0;
    F(i,:) = tmpF_mean';
end
fprintf('Done!\n');toc





