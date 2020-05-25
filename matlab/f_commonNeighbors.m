function F = f_commonNeighbors(D)
    % return a dense matrix with element (i,j) = products of the degree of
    % node i and node j
    
    F = D^2-diag(diag(D^2));
    
  