function [U,idxU, centersU,  eigF_topd, values, max_values] = BIN_decompose_F(fhandle, eigA_topl, vecA_topl, dim, bins)
% Spectral decomposition of F ~= U*D*U', where rows in U are given by the centroids of a
% k-means clustering into a number of bins.
% We assume F is a polynomial of the adjacency matrix A, specified by
% fhandle, and thus use eigeninformation of A to determine eigeninformation
% of F.

eigF_l = fhandle(eigA_topl);
[~,j] = sort(abs(eigF_l),'descend');
eigF_topd = eigF_l(j);
eigF_topd = eigF_topd(1:dim); % the top-d eigenvalues of F.
vecF_topd = vecA_topl(:,j); % the associated eigenvectors.
vecF_topd = vecF_topd(:,1:dim);

U = vecF_topd; 
% Now k-means on U to bin them into groups
% [idxU,centersU] = kmeans(U, bins,'Replicates',3);
% % values indicates a matrix with F values between corresponding bins.
% values = centersU*diag(eigF_topd)*centersU';
% max_values = max(max(values)); 
% values = values/max_values; % Rescaling for numerical stability of Newton's method.

s = sign(eigF_topd);
scaling = sqrt(abs(eigF_topd));
U_resc = U*diag(scaling);
% Now k-means on U to bin them into groups
[idxU,centersU] = kmeans(U_resc, bins,'Replicates',3);
values = centersU*(centersU*diag(s))';
max_values = max(max(values));
values = values/max_values;
