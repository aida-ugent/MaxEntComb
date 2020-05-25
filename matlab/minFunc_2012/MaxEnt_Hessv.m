function Hv = MaxEnt_Hessv(v, x, M, F, nF)
% Computing the product of H(x)*v, where H(x) is a diagonal approximation of the
% Hessian at point x.

n = length(M); % reduced model length
s = cellfun(@times,F,num2cell(x(end-nF+1:end)'),'uni',false); % multiplying F's by their lambda
s = sum(cat(3,s{:}),3); % summing over all F's
A1 = x(1:n) + x(n+1:2*n)' + s;
A1 = exp(A1);
A = (A1./(1+A1));
A(isnan(A))=1; % replacing overflows by probability 1.
A = M.*A;

% Hessian
A = A./(1+A1);
C = cellfun(@(x) x.*x.*A, F, 'uni', false);
H = [sum(A,2); sum(A,1)'; cellfun(@(x)sum(sum(x)),C)'];

Hv = H.*v;

end