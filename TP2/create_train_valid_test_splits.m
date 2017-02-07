function [XA XV XT  ] = create_train_valid_test_splits( X )
%CREATE_TRAIN_VALIDE_TEST_SPLITS Summary of this function goes here
%   Detailed explanation goes here
indices = randperm(size(X,2));
pourcentages = floor([0.6*size(indices,2) 0.2*size(indices,2) 0.2*size(indices,2)]);

size(X);
XA = indices(:, 1:pourcentages(1));
XV = indices(:, pourcentages(1):pourcentages(2));
XT = indices(:, pourcentages(2),:);


end

