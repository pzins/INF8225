load 20news_w100
whos
n = 4
m = size(newsgroups, 2)
o = ones(1, m)
i = 1:m
j = newsgroups
Y = sparse(i, j, o, m, n)

Theta = rand(4, 101)-0.5;
X = documents;
X = [X; ones(1,16242)];
taux_dapprentissage = 0.0005;


[XA, XV, XT] = create_train_valid_test_splits(X);
break
while ~converged
    
end

