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

[XA, XV, XT, YA, YV, YT] = create_train_valid_test_splits(X, Y);

%{
converged = false;
yixi = YA' * XA';
logV = -10e10;
precision = [0 0]

while ~converged  
    oldLogV = logV
    
    % calcul de la log vraisemblance
    Z = sum(exp(Theta * XA)); %Z (ou partie droite du calcul)
    left = sum((YA * Theta)' .* XA); %partie gauche du calcul
    logV = sum(left-log(Z))
    
    
    oldPrecision = precision;
    % calcul de P(Y|X)
    P_Y_sachant_X = exp(Theta*XA) ./ [Z;Z;Z;Z];
    % calcul du gradient
    right = P_Y_sachant_X * XA'; %partie droite du calcul
    gradient = yixi-right;


    % calculer precision ensemble d'apprentissage
    precisionA = get_precision(XA,YA, Theta);
    fprintf('precision ensemble de''apprentissage : %f\n', precisionA);

    % calculer precision ensemble validation
    precisionV = get_precision(XV,YV, Theta);
    fprintf('precision ensemble de validation : %f\n', precisionV);

    % update precision
    precision = [precisionA precisionV];
    
    % update theta
    Theta = Theta + taux_dapprentissage * gradient;
    
    % verification de la condition d'arrêt
    if (abs(oldPrecision(2) - precision(2)) < 0.001),
        converged = 1
    end
   
    pause
end

% calculer precision ensemble test
precisionT = get_precision(XT,YT, Theta);
fprintf('precision ensemble de test: %f\n', precisionT);
%}

% mini-batch
Theta = rand(4,101) - 0.5;
converged = false;
logV = -10e10;
precision = -10;
taux_dapprentissage = 0.0001;

t = 1;
while ~converged  
    oldLogV = logV
    
    [X_batch, Y_batch] = get_mini_batch(XA, YA, 20)
    taux_dapprentissage = t;
    for i = 1:size(X_batch,2),
        oldprecision = precision;
        % calcul de la log vraisemblance
        Z = sum(exp(Theta * X_batch{:,i})); %Z (ou partie droite du calcul)
        left = sum((Y_batch{i} * Theta)' .* X_batch{:,i}); %partie gauche du calcul
        logV = sum(left-log(Z));
        
        % calcul de P(Y|X)
        P_Y_sachant_X = exp(Theta*X_batch{:,i}) ./ [Z;Z;Z;Z];
        % calcul du gradient
        right = P_Y_sachant_X * X_batch{:,i}'; %partie droite du calcul
        yixi = Y_batch{i,:}' * X_batch{:,i}';
        gradient = (yixi-right)./size(X_batch{:,i},2);

        % update theta
        Theta = Theta + taux_dapprentissage * gradient;

       
        % calculer precision sur le mini batch
        precision = get_precision(XV,YV, Theta);
        fprintf('precision ensemble de validation: %f\n', precision);
    
    
        % verification de la condition d'arrêt
        if (abs(oldprecision - precision) < 0.0001),
            converged = 0
        end
    end
    
    t = t + 1;
    pause
end

% calculer precision ensemble test
precisionT = get_precision(XT,YT, Theta);
fprintf('precision ensemble de test: %f\n', precisionT);


