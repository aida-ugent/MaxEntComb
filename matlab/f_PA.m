function [idxPA, valuesPA] = f_PA(D)
    % Returns an indices vector, and unique values matrix for F =
    % Preferential Attachment.
    [u,~,j ] = unique(sum(D,2));
    idxPA = j;
    valuesPA = full(u*u');
    valuesPA = valuesPA / max(max(valuesPA)); % Renormalizing for numerical stability.