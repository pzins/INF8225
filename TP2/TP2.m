load 20news_w100
whos;
n = 4;
m = size(newsgroups, 2);
o = ones(1, m);
i = 1:m;
j = newsgroups;
Y = sparse(i, j, o, m, n);

Theta = rand(4, 101)-0.5;
X = documents;
X = [X; ones(1,16242)];
taux_dapprentissage = 0.0005;
possibleY = eye(n);

[XA, XV, XT] = create_train_valid_test_splits(X);
converged = false;
size(X)

% r = (Y*Theta).*X';
% sum(sum(r))
log(sum(diag((Theta*X)*Y)))

logVraisemblance = sum(sum(((Y * Theta) .* X')') - log(sum(exp(possibleY * Theta * X))));

% size(X)
% (Theta'*X')

% sum(sum(Theta*X))
% sum(sum(Theta .* X'))
break
while ~converged
%    ds le premier mult d'abord par Y permet de ne pas faire pr ts les Y
%   ds le second, on fait une somme sur tt les Y dc on fait juste Theta*X

    logV = sum(sum((Y*Theta) .* X) - log(sum(exp(Theta*X))))
%         logVraisemblance = sum(sum(((YA * Theta) .* XA')') - log(sum(exp(possibleY * Theta * XA))));

end

