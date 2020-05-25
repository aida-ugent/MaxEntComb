function [ind_Cartesian, values_all_Reshaped] = BIN_CartesianBinning(indices_all, values_all)
% Finding the least restricting cartesian product of indices in
% indices_all, with reshaped values matrices to fit the 'larger' binning.
% Assumes indices_all and values_all are cells

% Finding the least restricting cartesian product of indices in indices_all
[unique_Combos,~,ind_Cartesian] = unique(cell2mat(indices_all),'rows','legacy');

% Reshaping the values to shapes that are matched to the cartesian product.
values_all_Reshaped = cell(1,length(values_all));
for i=1:length(values_all)
    loc = values_all{i};
    re = unique_Combos(:,i);
    values_all_Reshaped{i} = loc(re,re);
end

