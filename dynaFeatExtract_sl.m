% DynFV feature extraction
% INPUT: denseTrj[1xN cell] - cell variable contains dense trajectories
%                             for N identities; each cell is a Mx(2l+1)
%                             matrix for M trajectories of length l; each
%                             row is the formation [index x1 y1 x2 y2 ...]
%        train[1xNtrain]    - index of the training identities
%        test, gID          - not used in currunt version
%        camID[1xN]         - cam ID for each identity
%        endp[1xN cell]     - ending point relativly position for each traj
%        startP[1xN cell]   - starting point relativly position for each
%                             traj
%        imEdge[1xN cell]   - gray scale edge map for each identities
%        dynOpt             - options
%              .nr          - number of rows of the grid inside the bbox
%              .nc          - number of column of the grid inside the bbox
%              .PatchOverlap- 0/1: without/with overlap for the grid
%              .ncenter     - number of gaussians of the GMM model
%              .edgeMask    - 0/1: without/with background mask based on
%                             semantic edge map
%              .sl          - sliding window size
% OUTPUT: F         - DynFV features
%         codebook  - codebook of GMM model
% Author: Mengran Gou @ NEU robust systems lab 07/16/2016
%%
function [F,codebook] = dynaFeatExtract_sl(...
    denseTrj,train,camID,endp,startP,imEdge,dynOpt)
try
    vl_version
catch
    error('Cannot find vl_feat!')
end

%% Foreground mask based on mean edge image
if dynOpt.edgeMask
    uCam = unique(camID); % one mask per camera
    for c = 1:numel(uCam)
        idxC = find(camID==uCam(c));
        idxC_train = intersect(train,idxC);
        % filter out background based on edge mask
        meanEdge = cell2mat(imEdge(idxC_train));
        meanEdge = reshape(meanEdge,128,64,[]);
        meanEdge = mean(meanEdge,3);
        weightEdge{c} = meanEdge./sum(meanEdge(:));
        
        meanEdge(meanEdge<mean(meanEdge(:)))=0;
        meanEdge(meanEdge>0) = 1;
        se = strel('disk',5,4);
        tmpEdge = imdilate(meanEdge,se);
        tmpEdge([1 128],:) = 0;
        tmpEdge(:,[1 64]) = 0;
        tmpEdge = imfill(tmpEdge,'holes');
        meanEdge = tmpEdge;
        % meanEdge = imerode(tmpEdge,se);
        fgzone = find(meanEdge)';
        
        for i = 1:numel(idxC)
            tmpendP = endp{idxC(i)};
            %             tmpendP = denseTrj{idxC(i)}(:,end-1:end);
            intendP = round(tmpendP);
            intendP(intendP<1) = 1;
            intendP(intendP(:,1)>64,1)=64;
            intendP(intendP(:,2)>128,2)=128;
            endPidx = sub2ind([128 64],round(intendP(:,2)),round(intendP(:,1)));
            %         label_fg = ismember(endPidx,fgzone);
            
            %             tmpstartP = denseTrj{idxC(i)}(:,2:3);
            tmpstartP = startP{idxC(i)};
            intstartP = round(tmpstartP);
            intstartP(intstartP<1) = 1;
            intstartP(intstartP(:,1)>64,1)=64;
            intstartP(intstartP(:,2)>128,2)=128;
            startPidx = sub2ind([128 64],round(intstartP(:,2)),round(intstartP(:,1)));
            
            label_fg = ismember(endPidx,fgzone) & ismember(startPidx,fgzone);
            
            weightEndCell{idxC(i)} = weightEdge{c}(endPidx);
            weightStartCell{idxC(i)} = weightEdge{c}(startPidx);
            denseTrj{idxC(i)} = denseTrj{idxC(i)}(label_fg,:);
            endp{idxC(i)} = endp{idxC(i)}(label_fg,:);
            startP{idxC(i)} = startP{idxC(i)}(label_fg,:);
        end
    end
    weightEnd = cell2mat(weightEndCell');
    weightStart = cell2mat(weightStartCell');
    weight = max(weightEnd, weightStart);
else
    [numTrjPerI,~] = cellfun(@size, denseTrj,'UniformOutput',1);
    weight = ones(sum(numTrjPerI),1);
end

%% Preparing data

% label each trajectory with the gID it belongs to
[numTrjPerI,~] = cellfun(@size, denseTrj,'UniformOutput',1);
index = cumsum(numTrjPerI);
index = [0,index];
label_trajID = zeros(1,index(end));
for i = 1:numel(index)-1
    label_trajID(index(i)+1:index(i+1)) = i;
end
ind_train = ismember(label_trajID,train); % training trajs label

% assign spatial label for each traj based on its ending point position
warning off
nr = dynOpt.nr;
nc = dynOpt.nc;

normSz = [128 64]; % rxc
[~, BBox, ~] = GenerateGridBBox_numP(normSz, nr, nc, dynOpt.PatchOverlap);
BBox = ceil(BBox);
endp = cell2mat(endp');
% startP = cellfun(@(X) X(:,2:3),denseTrj,'uni',false);
startP = cell2mat(startP');
label_strip = zeros(numel(label_trajID),nr*nc);
for g = 1:nr*nc
    xv = [BBox(g,1),BBox(g,1),BBox(g,3),BBox(g,3),BBox(g,1)];
    yv = [BBox(g,2),BBox(g,4),BBox(g,4),BBox(g,2),BBox(g,2)];
    %     label_strip(:,g) = inpolygon(endp(:,1),endp(:,2),xv,yv);
    label_strip(:,g) = inpolygon(startP(:,1),startP(:,2),xv,yv);
end
label_strip = logical(label_strip);
num_strip = size(label_strip,2);


%% Normalization and sliding window partition

tmpTrj = cell2mat(denseTrj');
tmpTrj = tmpTrj(:,2:end);

% calculate the speed
tmpX = tmpTrj(:,1:2:end);
tmpY = tmpTrj(:,2:2:end);
velX = tmpX(:,2:end)-tmpX(:,1:end-1);
velY = tmpY(:,2:end)-tmpY(:,1:end-1);


% sliding window augment feature
clear tmpTrj tmpX tmpY
sl_win_m = dynOpt.sl;
if 1 % given the length of the short trajs, fully overlapped
    sl_win_m = dynOpt.sl;
    for sl = 1:numel(sl_win_m) % loop on different length
        sl_win = sl_win_m(sl);
        tmpTrj{sl} = zeros(size(velX,1)*(size(velX,2)-sl_win+1),sl_win*2,'single');
        label_sl{sl} = [];
        for slw = 1:size(velX,2)-sl_win+1 % loop on slides
            tmpTrj{sl}((slw-1)*size(velX,1)+1:slw*size(velX,1),1:2:end) = velX(:,slw:slw+sl_win-1);
            tmpTrj{sl}((slw-1)*size(velX,1)+1:slw*size(velX,1),2:2:end) = velY(:,slw:slw+sl_win-1);
            label_sl{sl} = [label_sl{sl}, slw*ones(1,size(velX,1))];
        end
        label_trajID_sl{sl} = repmat(label_trajID,1,size(velX,2)-sl_win+1);
        label_strip_sl{sl} = repmat(label_strip,size(velX,2)-sl_win+1,1);
        ind_train_sl{sl} = repmat(ind_train,1,size(velX,2)-sl_win+1);
        weight_sl{sl} = repmat(weight,size(velX,2)-sl_win+1,1);
    end
else % givin the numver of number segments per layer, no overlapping
    sh_num = dynOpt.sl;
    for sh = 1:numel(sh_num) % loop on different layers
        shn = sh_num(sh);
        sl_win = floor(size(velX,2)/shn);
        curP = 1;
        tmpTrj{sh} = zeros(size(velX,1)*shn,sl_win*2,'single');
        label_sl{sh} = [];
        for tmplabel = 1:shn % loop on segments
            tmpTrj{sh}((tmplabel-1)*size(velX,1)+1:tmplabel*size(velX,1),1:2:end) = velX(:,curP:curP+sl_win-1);
            tmpTrj{sh}((tmplabel-1)*size(velX,1)+1:tmplabel*size(velX,1),2:2:end) = velY(:,curP:curP+sl_win-1);
            label_sl{sh} = [label_sl{sh}, tmplabel*ones(1,size(velX,1))];
            curP = curP+sl_win;
        end
        label_trajID_sl{sh} = repmat(label_trajID,1,shn);
        label_strip_sl{sh} = repmat(label_strip,1,shn);
        ind_train_sl{sh} = repmat(ind_train,1,shn);
    end
end

for sl = 1:numel(tmpTrj)
    tmpfeature = tmpTrj{sl}';
    feature{sl} = tmpfeature;
end
clear tmpTrj;

%%      OPTIONAL: Whitening PCA (seems down grade the performance)
%         % feature is colomn wise
%         if ~iscell(feature)
%             feature = {feature};
%         end
%         for c = 1:numel(feature)
%             feature{c} = whitening(feature{c});
% % %             feature = feature(1:round(size(feature,1)/2),:); % only keep partial feature
%         end
%%      Build GMM
% EM, learning GMM model for each strip/patch
disp('Begin to extract FV...')
LL = [];
for s = 1:num_strip
    for sl = 1:numel(sl_win_m)
        feature_tmp = feature{sl}(:,ind_train_sl{sl} & label_strip_sl{sl}(:,s)');
        subind = randsample(size(feature_tmp,2),min(size(feature_tmp,2),50000)); % subsample to speed up
        feature_train = feature_tmp(:,subind);
        if size(feature_train,2) < dynOpt.ncenter
            continue;
        end
        [means{sl}{s}, covariances{sl}{s}, priors{sl}{s},LL(sl,s),~] = vl_gmm(feature_train,dynOpt.ncenter,'NumRepetitions',1);
    end
end
%% Fisher vector encoding
for i = 1:numel(denseTrj)
    clear tmpF;
    for sl = 1:numel(sl_win_m)
        tmpF{sl} = zeros(dynOpt.ncenter*size(feature{sl},1)*2,num_strip);
    end
    for s = 1:num_strip
        for sl = 1:numel(sl_win_m)
            tmpFeature = feature{sl}(:, label_trajID_sl{sl}==i & label_strip_sl{sl}(:,s)');
            if isempty(tmpFeature)
                continue;
            end
            % encoding
            tmpF{sl}(:,s) = vl_fisher(tmpFeature, means{sl}{s}, covariances{sl}{s}, priors{sl}{s},'Normalized','SquareRoot');
        end
    end
    tmpF = cell2mat(tmpF');
    F(i,:) = tmpF(:);
end
codebook.means = means;
codebook.cov = covariances;
codebook.prior = priors;
end


function featW = whitening(feat)
feat = feat';
feat = bsxfun(@minus, feat,mean(feat,2)); % remove mean
[pcaEigVec,pcaCenter,pcaEigVal] = pca(feat); % pca
featW = diag(pcaEigVal.^-0.5)*pcaEigVec'*feat; % whitening
featW = featW(1:round(size(featW,1)/2),:); % only keep partial feature
featW = featW';
end
