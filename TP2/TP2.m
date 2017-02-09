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

% calcul de la log vraisemblance
Z = sum(exp(Theta * X)); %Teta * X pr faire le produit coeff x instance pr chaque classe
% on retrouve une matrice 4x16242 avc un score pr chaque classe pr chaque
% instance puis on somme sur les classes
left = sum((Y * Theta)' .* X); %Y*Theta => coeff corrspondant ) la bonne classe à chaque position du Y, on transpose pr avoir 101x16242 comme X
% sum pr avoir pr chaque instance une valeur 
% sum(left-log(Z))

% calcul de P(Y|X)
up = exp(Theta*X); %numerateur
% Z = denominateur
PYsX = up ./ [Z;Z;Z;Z];


yixi = Y' * X';

% sum(sum(log(Y*P_Y_sachant_X))) %test mais en faite pas encore ca
a = sum((Y * Theta)' .* X);
sum(a-log(Z));

% logVraisemblance = sum(sum(((Y * Theta) .* X')') - log(sum(exp(possibleY * Theta * X))))
% logVraisemblance_ = sum(sum(((Y * Theta) .* X')') - log(sum(exp(Theta*X))));

left = Y' * X';
right = PYsX * X';
a = left-right;


Z = repmat(sum(exp(Theta * X)),4,1);
esperance = ((exp(Theta * X)./Z)' )' * X';
gradient = yixi - esperance;
min(min(a==gradient))

break    
while ~converged
%    ds le premier mult d'abord par Y permet de ne pas faire pr ts les Y
%   ds le second, on fait une somme sur tt les Y dc on fait juste Theta*X

    logV = sum(sum((Y*Theta) .* X) - log(sum(exp(Theta*X))))
%         logVraisemblance = sum(sum(((YA * Theta) .* XA')') - log(sum(exp(possibleY * Theta * XA))));

end

