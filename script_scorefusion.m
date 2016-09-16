% clc
clear
%%
dataname = 'PRID';
profix = '_allbk_35';
Parti = 'Random';%'DVR';%
ML = 'unsuper';
resFolder = 'results_BMVC';%'results';

feature = {'dyn3sl2slf'}; %'dyn3sl', 'HOG3D', 'LBP', 'Hist', 'Color', 'LDFVmaskcolor','LDFVmaskedge'
Type = 'unsuper';%'Tanh';%'Tanh';%'LikelihoodRatio';
weights = [1 1 1];
finalCMC = [];

load(['Dataset/' dataname '_Images_Tracklets_l15' profix],'gID');
load(['Feature/' dataname '_Partition_' Parti profix]);

for f = 1:numel(feature)
    results = dir([resFolder '/Result_' dataname '_' ML '_' dataname '_' feature{f} '*_' Parti '_*.mat']);
    results = results(1);
    fprintf('Feature to fuse %s\n',results.name);
    load([resFolder '/' results.name]);
    Methods{f} = Method;
    num_rep(f) = AlgoOption.num_rep;
end

for idp = 1:numel(Partition)
    trainTabels = cell(1,numel(feature));
    testTabels = cell(1,numel(feature));
    gID_train = gID(Partition(idp).idx_train);
    trainMask = pdist2(gID_train',gID_train');
    trainMask = trainMask==0;
    for rep = 1:max(num_rep)
        for f = 1:numel(feature)
            testTabels{f} = Methods{f}{idp,min(num_rep(f),rep)}.pdtable.testTabel;
            trainTabels{f} = Methods{f}{idp,min(num_rep(f),rep)}.pdtable.trainTabel;
%             testTabels{f} = pdtable.testTabel;
%             trainKdis{f} = pdtable.trainKdis;
%             testKdis{f} = pdtable.testKdis;
        end
        if strcmp(Type,'unsuper')
            FusedTable = zeros(size(testTabels{1}));
            for f = 1:numel(feature)
                % normalize to [0,1]
                testTabels{f} = (testTabels{f}-min(testTabels{f}(:)))./max(testTabels{f}(:));
                FusedTable = FusedTable + weights(f).*testTabels{f};
%                 FusedTable = FusedTable.*testTabels{f}.*weights(f);
            end
        else 
            [parameters, wgts, FusedTable, learningTables_Trans] = ...
            fusion.LearningandFusion(trainTabels, testTabels, trainMask, Type);
        end
        % generate CMC
        gID_test = gID(Partition(idp).idx_test);
        ix_test_gallery = Partition(idp).ix_test_gallery;
        for ix = 1:size(Partition(idp).ix_test_gallery,1)
            ID_gal = find(ix_test_gallery(ix,:)==1);
            ID_prob = find(ix_test_gallery(ix,:)==0);
            r = [];
            for g = 1:numel(ID_prob)
                [~,sortx] = sort(FusedTable(ID_prob(g),ID_gal));
                %             [~,sortx] = sort(testKdis{1}(ID_gal(g),ID_prob));
                r(g) = find(gID_test(ID_gal(sortx))==gID_test(ID_prob(g)));
            end
            rank{idp}(ix,:) = r;
        end
        %display
        [a, b] = hist(rank{idp}',1:sum(ix_test_gallery(1,:)==1));
        if min(min(double(ix_test_gallery)))<0
            a = cumsum(a)./repmat(sum(ix_test_gallery==-1,2)', size(a,1),1);
        else
            a = cumsum(a)./repmat(sum(ix_test_gallery==0,2)', size(a,1),1);
        end
        if size(a,1) ~= size(ix_test_gallery,1)
            a = a';
        end
%         for itr =1: size(a,1)
%             rr(itr,:)= [a(itr,1) a(itr,5) a(itr,10) a(itr,20) a(itr,25)];
%             display(['itration ' num2str(itr) ' Rank 1 5 10 20 25 accuracy ===>' num2str(rr(itr,:)) ' ====']);
%         end
%         display(num2str([mean(rr,1)]));
        finalCMC = [finalCMC;a];
    end
end
meanCMC = mean(finalCMC(1:1:end,:),1); % fix the fisrt camera as probe

% CalculatePUR(meanCMC,sum(ix_test_gallery(1,:)))*100
resshow = [meanCMC([1 5 10 20])*100 CalculatePUR(meanCMC,sum(ix_test_gallery(1,:)))*100]
    
