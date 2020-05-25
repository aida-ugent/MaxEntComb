function [idxRA, valuesRA] = f_RA(degrees, eigA_topd, vecA_topd, bins)
    % Returns an indices vector, and unique values matrix for F =
    % Resource Allocation Index.
    
    n = length(vecA_topd);
    dim = length(eigA_topd);
    idxRA = zeros(n,1);
    
    % Degree 0 nodes
    ind0 = find(degrees==0);
    idxRA(ind0) = bins; % The degree 0 nodes go into the last bin.
    
    % Degree >= 1 nodes
    ind1 = find(degrees>0);
    l = length(ind1);
    %D = diag(1./degrees(ind1));
    D = spdiags(1./degrees(ind1), 0, l, l);
    vecA_topd = vecA_topd(ind1,:);
    
    %Vtilde = vecA_topd*sqrtm(diag(eigA_topd)*vecA_topd'*D*vecA_topd*diag(eigA_topd));
    Vtilde = vecA_topd*sqrtm(spdiags(eigA_topd,0,dim,dim)*vecA_topd'*D*vecA_topd*spdiags(eigA_topd,0,dim,dim));
    [idx_k,centers_k] = kmeans(Vtilde, max(1,bins-1),'Replicates',5);
    idxRA(ind1) = idx_k;
    valuesSub = centers_k*centers_k';
    valuesRA = [valuesSub zeros(max(1,bins-1),1); zeros(1,max(1,bins-1)) 0];
    valuesRA = valuesRA / max(max(valuesRA)); % Renormalizing for numerical stability.