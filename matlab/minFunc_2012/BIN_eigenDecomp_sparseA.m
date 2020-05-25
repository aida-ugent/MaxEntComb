function [eigA_topl, vecA_topl] = BIN_eigenDecomp_sparseA(A, dim)

% Computing the top l >= dim (measured in abs. value) eigendecomposition of a
% sparse matrix A, such that the top l eigenvalues contain atleast 'dim' positive
% eigenvalues (see paper)

% First try to compute l = 2*dim, and increase if needed.
l_all = (2:10)*dim; % sufficient in practice
n = length(A);

for k=1:length(l_all)
    l_try = min(l_all(k),n);
    [vecA_topl,eigA_topl] = eigs(A,l_try,'lm'); % Computing the top-l largest (in abs. value) eigenvectors/values.
    eigA_topl = diag(eigA_topl);
    [~,ind] = sort(abs(eigA_topl),'descend'); 
    eigA_topl = eigA_topl(ind); % Sorted eigenvalues by abs value.
    vecA_topl = vecA_topl(:,ind);
    
    if length(find(eigA_topl>0)) >= dim
        % Enough eigenvalues found
        break
    end
end