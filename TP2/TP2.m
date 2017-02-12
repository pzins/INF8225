load 20news_w100
whos
n = 4;
m = size(newsgroups, 2);
o = ones(1, m);
i = 1:m;
j = newsgroups;
Y = sparse(i, j, o, m, n);

X = documents;
X = [X; ones(1,16242)];

[XA, XV, XT, YA, YV, YT] = create_train_valid_test_splits(X, Y);

%initial values
logV = [];
precisions = [0 0];
taux_dapprentissage = 0.0005;   
Theta = rand(4, 101)-0.5;
converged = false;


yixi = YA' * XA'; %left part for the gradient part. value is constant
while ~converged  
    oldPrecisions = precisions;
    
    % compute log vraisemblance
    Z = sum(exp(Theta * XA)); %denominator
    numerator = sum((YA * Theta)' .* XA);
    logV = [logV sum(numerator-log(Z))];
    fprintf('\nLog vraisemblance : %f\n', logV(1,end));
    
    % compute P(Y|X)
    P_Y_sachant_X = exp(Theta*XA) ./ [Z;Z;Z;Z];
    % compute gradient
    right_part = P_Y_sachant_X * XA'; %right side of the formula
    gradient = yixi - right_part;

    % compute training set precision
    precisionA = get_precision(XA,YA, Theta);
    fprintf('precision ensemble de''apprentissage : %f\n', precisionA);

    % compute validation set precision
    precisionV = get_precision(XV,YV, Theta);
    fprintf('precision ensemble de validation : %f\n\n', precisionV);

    % update precision
    precisions = [precisions ; [precisionA precisionV]];
    
    % update theta
    Theta = Theta + taux_dapprentissage * gradient;
    
    % check convergence
    if (abs(oldPrecisions(end,end) - precisions(end,end)) < 0.0001),
        converged = 1;
    end
    
end

precisions = precisions(2:end, :); %remove initial values

% compute test set precision
precisionT = get_precision(XT,YT, Theta);
fprintf('precision ensemble de test: %f\n', precisionT);
precisions = [precisions repmat(precisionT, length(precisions), 1)];

% graph with precisions
figure
title('Batch : precisions during gradient descent')
plot(1:length(precisions), precisions)
ylabel( 'precision')
xlabel('iterations')
legend('learning set', 'validation set', 'test set')

% graph with log vraisemblance
figure
title('Batch : log vraisemblance')
plot(1:length(logV), logV)
ylabel( 'log vraisemblance')
xlabel('iterations')



% mini-batch

%initial values
logV = [];
precisions = [0 0];
taux_dapprentissage = 0.0005;   
Theta = rand(4, 101)-0.5;
converged = false;
NB_mini_batch = 20;

t = 1;
while ~converged
    
    %compute mini-batch
    [X_batch, Y_batch] = get_mini_batch(XA, YA, NB_mini_batch);
    taux_dapprentissage = t;
    for i = 1:size(X_batch,2),
        oldprecision = precisions;
        
        % compute log vraisemblance
        Z = sum(exp(Theta * X_batch{:,i})); % denominator
        numerator = sum((Y_batch{i} * Theta)' .* X_batch{:,i});
        logV = [logV sum(numerator-log(Z))];
        fprintf('\nLog vraisemblance : %f\n', logV(1,end));

        
        % compute P(Y|X)
        P_Y_sachant_X = exp(Theta*X_batch{:,i}) ./ [Z;Z;Z;Z];
        
        % compute gradient
        right_part = P_Y_sachant_X * X_batch{:,i}'; %partie droite du calcul
        yixi = Y_batch{i,:}' * X_batch{:,i}'; %not constant for mini-batch (depend on mini-batch)
        gradient = (yixi-right_part)./size(X_batch{:,i},2);

        % update theta
        Theta = Theta + taux_dapprentissage * gradient;
    end
    
    % compute precision on learning set
    precisionA = get_precision(XA,YA, Theta);
    fprintf('precision ensemble d''apprentissage: %f\n', precisionA);
    
    % compute precision on validation set
    precisionV = get_precision(XV,YV, Theta);
    fprintf('precision ensemble de validation: %f\n', precisionV);

    % update precision
    precisions = [precisions ; [precisionA precisionV]];
    
    
    % verification de la condition d'arrêt
    if (abs(oldprecision(end,end) - precisions(end,end)) < 0.001),
        converged = 1;
    end
    
    t = t + 1;
    
    
end
precisions = precisions(2:end,:); %remove initial values

% compute test set precision
precisionT = get_precision(XT,YT, Theta);
fprintf('precision ensemble de test: %f\n', precisionT);
precisions = [precisions repmat(precisionT, length(precisions), 1)];

% graph with precisions
figure
title('Batch : precisions during gradient descent')
plot(1:length(precisions), precisions)
ylabel( 'precision')
xlabel('iterations')
legend('learning set', 'validation set', 'test set')

% graph with log vraisemblance
figure
title('Batch : log vraisemblance')
plot(1:length(logV), logV)
ylabel( 'log vraisemblance')
xlabel('iterations')

