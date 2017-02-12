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

Theta_save = rand(4, 101)-0.5;
Theta = Theta_save;


%BATCH
taux = [0.0001 0.0005 0.0008]

for k=1:3,
    %initial values
    logV = [];
    precisions = [0 0];
    taux_dapprentissage = taux(k);
    converged = false;
    
    %reset Theta but not random (the same for the three learning rate)
    Theta = Theta_save;



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
        if (abs(oldPrecisions(end,end) - precisions(end,end)) < 0.01),
            converged = 1;
        end

    end

    precisions = precisions(2:end, :); %remove initial values

    % compute test set precision
    precisionT = get_precision(XT,YT, Theta);
    fprintf('precision ensemble de test: %f\n', precisionT);
    precisions = [precisions repmat(precisionT, length(precisions), 1)];
    
    precisions_taux{k} = precisions;
    logV_taux{k} = logV;
end

% graph with precisions
figure
plot(1:length(precisions_taux{1}), precisions_taux{1}, 1:length(precisions_taux{2}), precisions_taux{2},'--', 1:length(precisions_taux{3}), precisions_taux{3},':')
title('Batch : precisions during gradient descent')
ylabel('precision')
xlabel('iterations')
legend('taux = 0.0001 - learning set', 'taux = 0.0001 - validation set', 'taux = 0.0001 - test set', ...
'taux = 0.0005 - learning set', 'taux = 0.0005 - validation set', 'taux = 0.0005 - test set', ...
'taux = 0.0008 - learning set', 'taux = 0.0008 - validation set', 'taux = 0.0008 - test set')


% graph with log vraisemblance
figure
plot(1:length(logV_taux{1}), logV_taux{1}, 1:length(logV_taux{2}), logV_taux{2}, 1:length(logV_taux{3}), logV_taux{3})
title('Batch : log vraisemblance')
ylabel('log vraisemblance')
xlabel('iterations')
legend('taux = 0.0001', 'taux = 0.0005', 'taux = 0.0008')

break


% MINI_BATCH

%initial values
logV = [];
mbprecisions = [0 0];
mbprecisions_mini_batch = [0 0];
Theta = Theta_save;
converged = false;
NB_mini_batch = 20;
taux_dapprentissage_factor = 50;
t = 1;
while ~converged
    
    %compute mini-batch
    [X_batch, Y_batch] = get_mini_batch(XA, YA, NB_mini_batch);
    taux_dapprentissage = t/taux_dapprentissage_factor;
    oldmbprecision = mbprecisions;
    
    for i = 1:size(X_batch,2),
        oldmbprecision_mini_batch = mbprecisions_mini_batch;
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
        
        % add precision on learning set
        precisionA = get_precision(XA,YA, Theta);
        % add precision on validation set
        precisionV = get_precision(XV,YV, Theta);
        mbprecisions_mini_batch = [mbprecisions_mini_batch; [precisionA precisionV]];

    end
    
    % compute precision on learning set
    precisionA = get_precision(XA,YA, Theta);
    fprintf('precision ensemble d''apprentissage: %f\n', precisionA);
    
    % compute precision on validation set
    precisionV = get_precision(XV,YV, Theta);
    fprintf('precision ensemble de validation: %f\n', precisionV);

    % update precision
    mbprecisions = [mbprecisions ; [precisionA precisionV]];
    
    % verification de la condition d'arrêt
    if (abs(oldmbprecision(end,end) - mbprecisions(end,end)) < 0.001),
        converged = 1;
    end
    
    t = t + 1;
    
    
end
mbprecisions = mbprecisions(2:end,:); %remove initial values
mbprecisions_mini_batch = mbprecisions_mini_batch(2:end, :); %remove initial values

% compute test set precision
precisionT = get_precision(XT,YT, Theta);
fprintf('precision ensemble de test: %f\n', precisionT);
mbprecisions = [mbprecisions repmat(precisionT, length(mbprecisions), 1)];
mbprecisions_mini_batch = [mbprecisions_mini_batch repmat(precisionT, length(mbprecisions_mini_batch), 1)];

% graph with precisions
figure
plot(1:length(mbprecisions), mbprecisions)
str = sprintf('Mini-batch : precisions during gradient descent (each epoch), taux : t/%d', taux_dapprentissage_factor)
title(str)
ylabel('precision')
xlabel('iterations')
legend('learning set', 'validation set', 'test set')

% graph with log vraisemblance
figure
plot(1:length(logV), logV)
str = sprintf('Mini-batch : log vraisemblance, taux : t/%d', taux_dapprentissage_factor)
title(str)
ylabel('log vraisemblance')
xlabel('iterations')

% graph with precisions_mini_batch
figure
plot((1:length(mbprecisions_mini_batch))/20, mbprecisions_mini_batch)
str = sprintf('Mini-batch : precisions during gradient descent (each iteration in mini-batch) , taux : t/%d', taux_dapprentissage_factor)
title(str)
ylabel('precision')
xlabel('iterations')
legend('learning set', 'validation set', 'test set')
 

