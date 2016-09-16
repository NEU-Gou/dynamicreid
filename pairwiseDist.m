function pdtable = pairwiseDist(Method,train,test)
% compute the pairwise distance based on the learned metric
A = Method{1}.P; % projection matrix
if strcmp(Method{1}.name,'oLFDA') || isempty(Method{1}.kernel)
    K_test = test{1}';
    K_train = train{1}';
else
    K_test = ComputeKernelTest(train{1},test{1},Method{1});
    K_train = ComputeKernel(train{1},Method{1}.kernel,Method{1});
end

if strcmp(Method{1}.name,'rankSVM') || strcmp(Method{1}.name,'unsuper')
    p_test = bsxfun(@times, K_test,A);
    p_train = bsxfun(@times, K_train,A);
else
    p_test = A*K_test;
    p_train = A*K_train;
end

pdtable.testTabel = pdist2(p_test',p_test');
pdtable.testKdis = pdist2(K_test',K_test');
% testTabel = exp(-testTabel);
pdtable.trainTabel = pdist2(p_train',p_train');
pdtable.trainKdis = pdist2(K_train',K_train');
% trainTabel = exp(-trainTabel);
