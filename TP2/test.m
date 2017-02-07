clc;
clear;
load 20news_w100;
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);

Theta = rand(4,101)-.5;
X = [documents ; ones(1,16242)];
taux_dapprentissage = 0.0005;


lastLogVraisemblance = -realmax;
[XA XV XT YA YV YT] = create_train_valid_test_splits2(X,Y);


converted = false;
break

possibleY = eye(n);
converged = false;
yixi = YA' * XA';
baprecision = [];
bvprecision = [];
blogVraisemblances = [];

sum(exp(possibleY * Theta * XA))