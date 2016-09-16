clear
% Example code for "Person Re-id in Appearance Impaired Scenarios"
%% Parameters setting
addpath(genpath('Assistant Code/'));

loadfeat = 0; %load precomputed feautre, if exist
savefeat = 0;
bkmask = 1; % indicator for background mask

algoname = 'unsuper'; % perform un-supervised experiment
% kname={'linear'};%{'gaussian-rbf'};
% dataset name
dname = 'PRID';
% set profix for appearance impaired datasets
profix = '_allbk_35';%'_allbk_89';
% Partition setting, directly using partition from DVR paper for
% iLIDSVID/PRID
partition_name = 'Random'; %'DVR';
% feature name : LDFV; HistLBP; ColorLBP; DynFV
featname = 'LDFV';
% Set channel function for LDFV
% mask          - LDFV
% maskT         - tLDFV
% maskcolor     - LDFV-color
% maskedge      - LDFV-edge
if strcmp(featname,'LDFV')
    hf_chn = @mask; %@mask; @maskcolor,@maskedge,@maskT
end
% set number of patches in the bounding box [6 14 75]
num_patch = 75;

dataname = [dname '_Images_Tracklets_l15' profix];
load(['Dataset/' dataname '.mat']);
load(['Feature/' dname '_Partition_' partition_name profix '.mat']);

% choose appearance feature
switch featname
    case {'LDFV'}
        appFeaturename = ['_LDFV' func2str(hf_chn) num2str(num_patch) 'Patch'];
    case {'HistLBP'}
        appFeaturename = ['_Hist' num2str(num_patch) 'Patch'];
    case {'ColorLBP'}
        appFeaturename = ['_ColorLBP'];
end

if bkmask
    load(fullfile('Feature',[dname '_semEdge' profix]));
else 
    meanEdge = [];
end

% default algorithm option setting
AlgoOption.name = algoname;
AlgoOption.func = algoname;
AlgoOption.dataname = dname;
AlgoOption.partitionname = partition_name;
AlgoOption.useDynamic = 0; %0----App; 1----Dynamic
if strcmp(featname,'LDFV') || strcmp(featname,'DynFV')
    AlgoOption.num_rep = 10; % number of repeat for each partition
else
    AlgoOption.num_rep = 1;
end

rng('default');

t1 = clock;
timstr =[ num2str(int32(t1(1))) '_'  num2str(int32(t1(2))) '_'  num2str(int32(t1(3)))...
    '_'  num2str(int32(t1(4))) '_'  num2str(int32(t1(5))) '_'  num2str(int32(t1(6)))];
%%

for idx_partition=1:length(Partition) % partition loop
    for rep = 1:AlgoOption.num_rep % repeat loop
        fprintf('============== Partition %d -- rep %d ==============\n',idx_partition,rep);
        idx_train = Partition(idx_partition).idx_train ;
        idx_test = Partition(idx_partition).idx_test ;
        ix_test_gallery =Partition(idx_partition).ix_test_gallery;
        %% Feature extraction
        if strcmp(featname,'DynFV') % Using DynFV feature
            AlgoOption.dynOpt.nr = 6; % number of patches along the row (y) dimension
            AlgoOption.dynOpt.nc = 3; % number of patches along the col (x) dimension
            AlgoOption.dynOpt.PatchOverlap = 1; % Generated grids w/wo overlapping
            AlgoOption.dynOpt.ncenter = 12; % number of GMM learned
            AlgoOption.dynOpt.edgeMask = bkmask; % mask out the background by mean edge
            AlgoOption.dynOpt.sl = [5 9 14];
            
            savename = ['Result_' dname '_' AlgoOption.name '_' dname '_dyn' num2str(numel(AlgoOption.dynOpt.sl)) 'sl' profix '_' partition_name '_' timstr '.mat'];
            display(savename);
            
            if exist(['Feature/' dname '_dyn' num2str(numel(AlgoOption.dynOpt.sl)) 'sl' profix '.mat'],'file') && loadfeat
                load(['Feature/' dname '_dyn' num2str(numel(AlgoOption.dynOpt.sl)) 'sl' profix '.mat']);
                tmpFeat = normc_safe(FeatDyn_Keep{idx_partition,rep}');
                %                     load(fullfile('Feature',[fname '_semEdge' profix]));
            else
                tic
                if ~exist('endP','var') || ~exist('startP','var')
                    endP = cellfun(@(x) x(:,end-1:end),denseTrj,'uni',0);
                    startP = cellfun(@(x) x(:,2:3),denseTrj,'uni',0);
                end
                [feat_dyn,dynwords] = dynaFeatExtract_sl(denseTrj,idx_train,camID,endP,startP,meanEdge,AlgoOption.dynOpt);
                toc;
                FeatDyn_Keep{idx_partition,rep} = feat_dyn;
                tmpFeat = normc_safe(feat_dyn');
            end
            feat_dyn = tmpFeat';
            Feature = feat_dyn;
        else
            Featurename = [dname appFeaturename profix];
            if loadfeat
                load(fullfile('Feature',Featurename));
                tmpFeat = normc_safe(FeatApp_Keep{idx_partition,rep}');
            else
                savename = ['Result_' dname '_' AlgoOption.name '_' Featurename '_' partition_name '_' timstr '.mat'];
                display(savename);
                switch featname
                    case {'LDFV'}
                        FeatureAppearence = appFeatExtract_FV(I, idx_train, camID, num_patch, hf_chn, meanEdge);
                        FeatApp_Keep{idx_partition,rep} = FeatureAppearence;
                        tmpFeat = normc_safe(FeatApp_Keep{idx_partition,rep}');
                    case {'HistLBP'}
                        FeatureAppearence = appFeatExtract_HistLBP(I, idx_train, camID, num_patch,meanEdge);
                        FeatApp_Keep{idx_partition,rep} = FeatureAppearence;
                        tmpFeat = normc_safe(FeatApp_Keep{idx_partition,rep}');
                    case {'ColorLBP'}
                        FeatureAppearence = appFeatExtract_ColorLBP(I, idx_train, camID, num_patch,meanEdge);
                        FeatApp_Keep{idx_partition,rep} = FeatureAppearence;
                        tmpFeat = normc_safe(FeatApp_Keep{idx_partition,rep}');
                end
            end
            feat_app = tmpFeat';
            Feature = feat_app;
        end
        
        % cell???
        if ~iscell(Feature)
            tmpF{1} = Feature;
            Feature = tmpF;
        end
        % normalize
        feature_set = cell(1,numel(Feature));
        train = cell(1,numel(Feature));
        test = cell(1,numel(Feature));
        algo = cell(1,numel(Feature));
        for c = 1:numel(Feature)
            if numel(gID)~=size(Feature{c},1) % make sure the feature is N-by-d
                Feature{c} = double(Feature{c})';
            end
            feature_set{c} = Feature{c};
            train{c} = feature_set{c}(idx_train,:); % training set
            test{c} = feature_set{c}(idx_test,:); % test set
        end
        %% Train
        for c = 1:numel(train)
            switch AlgoOption.func % possible supervised learning here
                case {'unsuper'}
                    algo{c}.name = 'unsuper';
                    algo{c}.Prob = [];
                    algo{c}.Dataname = AlgoOption.dataname;
                    algo{c}.kernel = [];
                    algo{c}.Ranking = [];
                    algo{c}.Dist = [];
                    algo{c}.Trainoption=AlgoOption;
                    algo{c}.P = ones(size(train{1},2),1);
            end
        end
        %% Test
        % generate CMC
        if strcmp(AlgoOption.func,'unsuper')
            confusionMat = pdist2(test{1},test{1});
            gID_test = gID(idx_test);
            for ix = 1:size(ix_test_gallery,1)
                ID_gal = find(ix_test_gallery(ix,:)==1);
                ID_prob = find(ix_test_gallery(ix,:)==0);
                tmpr = [];
                for g = 1:numel(ID_prob)
                    [~,sortx] = sort(confusionMat(ID_prob(g),ID_gal));
                    tmpr(g) = find(gID_test(ID_gal(sortx))==gID_test(ID_prob(g)));
                end
                r(ix,:) = tmpr;
            end
            dis = confusionMat;
        end
        pdtable = pairwiseDist(algo,train,test);
        %% Show the result
        [a, b] = hist(r',1:sum(ix_test_gallery(1,:)==1));
        if min(min(double(ix_test_gallery)))<0
            a = cumsum(a)./repmat(sum(ix_test_gallery==-1,2)', size(a,1),1);
        else
            a = cumsum(a)./repmat(sum(ix_test_gallery==0,2)', size(a,1),1);
        end
        if size(a,1) ~= size(ix_test_gallery,1)
            a = a';
        end
        for itr =1: size(a,1)
            rr(itr,:)= [a(itr,1) a(itr,5) a(itr,10) a(itr,20) a(itr,25)];
            display(['itration ' num2str(itr) ' Rank 1 5 10 20 25 accuracy ===>' num2str(rr(itr,:)) ' ====']);
        end
        %% Save the result
        CMC{idx_partition,rep} = a;
        PUR(idx_partition,rep) = CalculatePUR(mean(a,1),sum(ix_test_gallery(1,:)))*100;
        display(num2str([mean(rr(:,1:4),1)*100 PUR(idx_partition,rep)]));
        Method{idx_partition, rep} = algo{1};
        for c = 1:numel(algo)
            tmpP = algo{c}.P;
        end
        Method{idx_partition, rep}.P = tmpP;
        Method{idx_partition, rep} = algo{1};
        Method{idx_partition, rep}.Prob = a;
        Method{idx_partition, rep}.Ranking = r;
        Method{idx_partition, rep}.Dist = dis;
        Method{idx_partition, rep}.pdtable = pdtable;
        save(savename, 'AlgoOption','Method','CMC','PUR');
    end
end


if savefeat
    if exist('FeatApp_Keep','var') && ~isempty(FeatApp_Keep{1})
        save(['Feature/' dname appFeaturename profix '.mat'], 'FeatApp_Keep','-v7.3');
    end
    if exist('FeatDyn_Keep','var') && ~isempty(FeatDyn_Keep{1})
        save(['Feature/' dname '_dyn' num2str(numel(AlgoOption.dynOpt.sl)) 'sl' profix], 'FeatDyn_Keep','-v7.3');
    end
end
