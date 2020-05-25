 function [x] = MaxEnt_Newton(D, ind_Cartesian, values_all_Reshaped, Gradtol)
% This code assumes the (symmetric) F matrix is factorized F ~= U*U', where U is determined by bins,
% ind indicates the resp. bin of a node, and values indicates a matrix with
% F values between corresponding bins.
% Assumes A is symmetric (undirected graphs).
degrees = sum(D,2);
b = max(ind_Cartesian);
n = length(D);
nF = length(values_all_Reshaped); % the number of global constraints

% Creating empty cells. 
bins = cell(1, b); % 
deg_u = cell(1, b); % unique degrees in a bin 
jr = cell(1, b); % a reference pointing towards each unique degree in a bin
occur = cell(1, b); % the occurences for each unique degree in a bin

l = zeros(b,1); % the number of unique degrees in a bin
for i=1:b
    bins{i} = find(ind_Cartesian==i);
    [deg_u{i}, ~, jr{i}] = unique(degrees(bins{i}));
    l(i) = length(deg_u{i});
    occur{i} = zeros(l(i),1);
    for k=1:l(i)
        occur{i}(k)=length(find(jr{i}==k));
    end 
end

du = cell2mat(deg_u'); % aligning all unique degrees
occAll = cell2mat(occur'); % aligning all occur cells
du = occAll.*du/n; % unique degrees times their occurences
M = (occAll*occAll'-diag(occAll))/n; % Occurence Matrix, not counting diagonal probs.
nred = length(occAll);

% Repeating F values inside each bin, by the number of occ's.
% Also computing c_F.
F = cell(1,nF);
c_F = zeros(nF,1);
[i,j] = find(triu(D,1)>0);
i = ind_Cartesian(i);
j = ind_Cartesian(j);
for k=1:nF
F{k} = repelem(values_all_Reshaped{k},l,l);
current = values_all_Reshaped{k};
temp = sub2ind(size(current),i,j);
c_F(k) = 2*sum(current(temp))/n;
end


f = @(x)objectiveAll(x, M, du, F, c_F, nF);
x0 = zeros(2*nred+nF,1);
addpath(genpath(pwd))
options.optTol = Gradtol;
options.MaxIter = 20000;
options.MaxFunEvals = 20000;
options.progTol = 10^-12; % default 10^-9;
options.Method = 'newton';
options.Display = 'final';
[x] = minFunc(f,x0,options);

% Recovering full x lambda parameters.
n=length(D);
y = zeros(2*n+nF,1); % y(1:n) are row, y(n+1:2*n) are column 
cuml = cumsum(l);
for i=1:b
    xbin = x(cuml(i)-l(i)+1:cuml(i));
    expand = xbin(jr{i});
    y(bins{i}) = expand;
    y(bins{i}+n) = expand;
end
y(end-nF+1:end) = x(end-nF+1:end);
x=y;