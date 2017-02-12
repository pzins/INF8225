function [ X_ Y_ ] = get_mini_batch( X, Y, n )
%GET_MINI_BATCH Summary of this function goes here
%   Detailed explanation goes her


indices = randperm(size(X,2));

tmp = repmat(ceil(size(X,2)/n), 1, n-1);
reste = size(X,2) - tmp(1,1) * (n-1);
tmp(n) = reste;

Y_random = Y(indices,:);
X_random = X(:,indices);

Y_ = mat2cell(Y_random, tmp, size(Y,2));
X_ = mat2cell(X_random, size(X,1), tmp);

end

