function test(tg, tr_e, te_e, tr_pred, te_pred, dim, bins, weights)
    % Main function to run the MaxentCombined model and obtain link
    % predictions. 
    % To evaluate with EvalNE, first run: mcc -mv test.m 
    % this will generate a run_test.sh file which can be executed without
    % having matlab installed. EvalNE only needs the path to this script
    % Bins = 200  Dim = 20
    % Weights indicate the multiplicity for each high-order priximity of A
    % we start at A^2. E.g. weights[1]*A^2 + weights[2]*A^3 ...
    
    dim = str2double(dim);
    bins = str2double(bins);
    
    % Read the input data
    delim = ',';
    edges = dlmread(tg, delim);
    train_e = dlmread(tr_e, delim);
    test_e = dlmread(te_e, delim);
    
    % Compute the number of nodes in the train graph tg
    nodes = unique(reshape(edges,1,[]));
    n = size(nodes, 2);
    
    % If the graph is 0-indexed, fix it
    if min(nodes) == 0
       edges = edges + 1;
       train_e = train_e + 1;
       test_e = test_e + 1;
    end
    
    % Compute de adj matrix and symmertize it
    idx = 1:n+1:n*n;
    A = sparse(edges(:,1), edges(:,2), 1, n,n);
    A = (A|A').*1;
    A(idx) = 0;

    % Compute Spectral Clustering of the F matrix.
    [eigA_topl, vecA_topl] = BIN_eigenDecomp_sparseA(A, dim);
    
    % Compute maxent with high order proximity A^2
    weights = [1,0];
	pows = 1:length(weights);
    pows = pows + 1;
    fhandle = @(x) sum(weights.*x.^pows, 2);
    %fhandle = @(x) x.^2; % a higher-order proximity
    [~, idxF1, centersU, eigF_topd, values_F1, max_values] = BIN_decompose_F(fhandle, eigA_topl, vecA_topl, dim, bins);
	
    % Compute maxent with high order proximity A^3
    weights = [0,1];
	pows = 1:length(weights);
    pows = pows + 1;
    fhandle = @(x) sum(weights.*x.^pows, 2);
    %fhandle = @(x) x.^2; % a higher-order proximity
    [~, idxF2, centersU, eigF_topd, values_F2, max_values] = BIN_decompose_F(fhandle, eigA_topl, vecA_topl, dim, bins);
	
    % Computing maxent with binned Resource Allocation Index
    eigA_topd = eigA_topl(1:dim);
    vecA_topd = vecA_topl(:,1:dim);
    degrees = sum(A,2);
    [idxRA, valuesRA] = f_RA(degrees, eigA_topd, vecA_topd, bins);

    % Compute low rank Preferential Attachment
    [idxPA, valuesPA] = f_PA(A);

    % Fit Reduced model with {F1,F2,...} global constraints
    % Block representation of the different F matrices.
    indices_all = {idxF1, idxF2, idxRA, idxPA}; % cell of indices, for each node (and each global constr.)
    values_all = {values_F1, values_F2, valuesRA, valuesPA}; % cell of diff. values, for each global constr.

    % Get Cart. Binning
    [ind_Cartesian, values_all_Reshaped] = BIN_CartesianBinning(indices_all, values_all);
    
    % Fit the maxent model using lbfgs solver
    tic;
    %cd minFunc_2012
    [x] = MaxEnt_lbfgs(A, ind_Cartesian, values_all_Reshaped, 10^-4);
    %cd ..
    toc;
    
    % Compute train predictions
    train_pred = zeros(1, size(train_e,1));
    for i=1:size(train_e,1)
        if train_e(i,1) == train_e(i,2)
            train_pred(i) = 0;
        else
            F1= values_F1(idxF1(train_e(i,1)),idxF1(train_e(i,2)));
			F2= values_F2(idxF2(train_e(i,1)),idxF2(train_e(i,2)));
			F_RA = valuesRA(idxRA(train_e(i,1)),idxRA(train_e(i,2)));
            F_PA = valuesPA(idxPA(train_e(i,1)),idxPA(train_e(i,2)));
            prob = x(train_e(i,1))+x(train_e(i,2)+n)+F1*x(end-3)+F2*x(end-2)+F_RA*x(end-1)+F_PA*x(end);
            prob = exp(prob)/(1+exp(prob));
            train_pred(i) = prob;
        end
    end
    csvwrite(tr_pred, train_pred');
    
    % Compute test predictions
    test_pred = zeros(1, size(test_e,1));
    for i=1:size(test_e,1)
        if test_e(i,1) == test_e(i,2)
            test_pred(i) = 0;
        else
            F1= values_F1(idxF1(test_e(i,1)),idxF1(test_e(i,2)));
			F2= values_F2(idxF2(test_e(i,1)),idxF2(test_e(i,2)));
            F_RA = valuesRA(idxRA(test_e(i,1)),idxRA(test_e(i,2)));
            F_PA = valuesPA(idxPA(test_e(i,1)),idxPA(test_e(i,2)));
            prob = x(test_e(i,1))+x(test_e(i,2)+n)+F1*x(end-3)+F2*x(end-2)+F_RA*x(end-1)+F_PA*x(end);
            prob = exp(prob)/(1+exp(prob));
            test_pred(i) = prob;
        end
    end
    csvwrite(te_pred, test_pred');
    
end
