load 20news_w100
whos
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


for k=1:4
    vector=zeros(1,4);
    vector(1,k)=1;
    P_Y(k,:)=exp(vector*Theta*X);
end
Z = sum(P_Y);
P_Y_sachant_X = P_Y ./ [Z;Z;Z;Z]
% sum(sum(P_Y_sachant_X .* X))

% Prob_y_sachant_x=P_Y;
% for k=1:length(Z)
%     Prob_y_sachant_x(:,k)=Prob_y_sachant_x(:,k)/Z(1,k);
% end

% Z = sum(exp(possibleY * Theta * X))
a=Z;
b=sum(exp(possibleY * Theta * X));
min(min(a==b));
% ----------------

% sum(sum(log(Y*P_Y_sachant_X))) %test mais en faite pas encore ca
a = sum((Y * Theta)' .* X)
sum(a-log(Z))

logVraisemblance = sum(sum(((Y * Theta) .* X')') - log(sum(exp(possibleY * Theta * X))))
logVraisemblance_ = sum(sum(((Y * Theta) .* X')') - log(sum(exp(Theta*X))));
break
left = Y' * X';

Z = sum(exp(Theta*X));
break



while ~converged
%    ds le premier mult d'abord par Y permet de ne pas faire pr ts les Y
%   ds le second, on fait une somme sur tt les Y dc on fait juste Theta*X

    logV = sum(sum((Y*Theta) .* X) - log(sum(exp(Theta*X))))
%         logVraisemblance = sum(sum(((YA * Theta) .* XA')') - log(sum(exp(possibleY * Theta * XA))));

end

