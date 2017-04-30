function [XA XV XT YA YV YT] = create_train_valid_test_splits( X ,Y)
%CREATE_TRAIN_VALIDE_TEST_SPLITS Summary of this function goes here
%   Detailed explanation goes here
indices = randperm(length(X)); %shuffle index
%size for each sets : learning 70%, validation 15%, test 15%
index = floor([0.7*length(indices) 0.85*length(indices)]); 

%learning, validation and test sets for X and Y
XA = X(:,indices(1:index(1)));
XV = X(:,indices(index(1):index(2)));
XT = X(:,indices(index(2):end));

YA = Y(indices(1:index(1)),:);
YV = Y(indices(index(1):index(2)),:);
YT = Y(indices(index(2):end),:);
end

