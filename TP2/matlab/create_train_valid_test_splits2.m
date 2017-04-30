function [ XA XV XT YA YV YT] = create_train_valid_test_splits2(X,Y)
    distribution = floor([0.7 0.15 0.15] * size(X,2));
    indices = randperm(size(X,2));
    
    sum = cumsum(distribution);
    b = sum - distribution + ones(1,size(distribution,2));
    
    XA = X(:, indices(b(1):sum(1)));
    XV = X(:, indices(b(2):sum(2)));
    XT = X(:, indices(b(3):sum(3)));
    YA = Y(indices(b(1):sum(1)), :);
    YV = Y(indices(b(2):sum(2)), :);
    YT = Y(indices(b(3):sum(3)), :);
end
