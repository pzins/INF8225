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
[XA XV XT YA YV YT] = create_train_valid_test_splits(X,Y);
possibleY = eye(n);
converged = false;
yixi = YA' * XA';
baprecision = [];
bvprecision = [];
blogVraisemblances = [];
%Approche par batch
while ~converged
    logVraisemblance = sum(sum(((YA * Theta) .* XA')') - log(sum(exp(possibleY * Theta * XA))));
    blogVraisemblances = [blogVraisemblances, logVraisemblance];
    converged = logVraisemblance - lastLogVraisemblance  < 0.1;
    lastLogVraisemblance = logVraisemblance;
    %Apprentissage
    Z = repmat(sum(exp(possibleY * Theta * XA)),4,1);
    esperance = ((exp(possibleY * Theta * XA)./Z)' * possibleY)' * XA';
    gradient = yixi - esperance;
    Theta = Theta + (taux_dapprentissage * gradient);
    baprecision = [baprecision, compute_precision(XA,YA,Theta)];
    bvprecision = [bvprecision, compute_precision(XV,YV,Theta)];
end
precisionTest = compute_precision(XT,YT,Theta);
fprintf('Precision sur lensemble de test = %f\n', full(precisionTest));


%Approche par mini-batch
Theta = rand(4,101) - 0.5;
deltaTheta = zeros(4,101);
alpha = 0.6;
batchSize = 568;
nbIteration = 1;
mblogVraisemblances = [];
lastLogVraisemblance = -realmax;
bmaprecision = [];
bmvprecision = [];
converged = false;
taux_dapprentissage = 0.0001;
temps = 1;
while ~converged
    [XBatch YBatch] = create_mini_batches(XA,YA,batchSize);
    taux_dapprentissage = 2/temps;
    for i = 1:size(XBatch,2)
        logVraisemblance = sum(sum(((YV * Theta) .* XV')') - log(sum(exp(possibleY * Theta * XV))));
        mblogVraisemblances = [mblogVraisemblances, logVraisemblance];
        converged = abs(logVraisemblance - lastLogVraisemblance)  < 0.001;
        if(converged)
            break;
        end
        lastLogVraisemblance = logVraisemblance;
        %Apprentissage
        Z = repmat(sum(exp(possibleY * Theta * XBatch{:,i})),4,1);
        esperance = ((exp(possibleY * Theta * XBatch{:,i})./Z) * XBatch{:,i}');
        yixi = YBatch{i,:}' * XBatch{:,i}';
        gradient = (yixi - esperance) ./batchSize;
        deltaTheta = alpha*deltaTheta + taux_dapprentissage * gradient;
        Theta = Theta + deltaTheta;
        bmvprecision = [bmvprecision, compute_precision(XV,YV,Theta)];
    end
    bmaprecision = [bmaprecision, compute_precision(XA,YA,Theta)];
    temps = temps + 1;
end
figure();
plot(1:size(baprecision,2),baprecision,1:size(bvprecision,2),bvprecision,(1:size(bmaprecision,2)) .* 20,bmaprecision,1:size(bmvprecision,2),bmvprecision);
title('Courbes d''apprentissage');
xlabel('Nombre d''itération');
ylabel('Précision(%)');
legend('Apprentissage(Batch)','Validation(Batch)','Apprentissage(miniBatch)','Validation(miniBatch)');

figure();
plot(1:size(blogVraisemblances,2),blogVraisemblances,1:size(mblogVraisemblances,2),mblogVraisemblances);
title('LogVraisemblance en fonction du nombre d''itération');
xlabel('Nombre d''itération');
ylabel('LogVraisemblance');
legend('Batch','MiniBatch');

precisionTest = compute_precision(XT,YT, Theta);
fprintf('Precision sur lensemble de test = %f\n', full(precisionTest));
Contact GitHub API Training Shop Blog About
© 2017 GitHub, Inc. Terms Privacy Security Status Help