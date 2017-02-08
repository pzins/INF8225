%% Énnoncé

load 20news_w100
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);
Theta = rand(4,101)-.5;
X = documents;
X = [X ; ones(1,16242)];
taux_dapprentissage =0.0005;
% [XA, XV, XT] = create_train_valid_test_splits(X);

%% Data Train

N=length(X);
set_data=randperm(N);

X_train=X(:,set_data(1:floor(0.7*N)));
Y_train=Y(set_data(1:floor(0.7*N)),:);

X_validation=X(:,set_data(floor(0.7*N)+1:0.85*N));
Y_validation=Y(set_data(floor(0.7*N)+1:0.85*N),:);

X_test=X(:,set_data(1+floor(0.85*N):N));
Y_test=Y(set_data(1+floor(0.85*N):N),:);

Distance=1500;


remember=1;
    % Calcul probabilité pour la partie test

for k=1:4
    vector=zeros(1,4);
    vector(1,k)=1;
    P_Y(k,:)=exp(vector*Theta*X_train);
end

Z=sum(P_Y);
Prob_y_sachant_x=P_Y;

for k=1:length(Z)
    Prob_y_sachant_x(:,k)=Prob_y_sachant_x(:,k)/Z(1,k);
end

%Calcul gradient
gradien=(X_train*Y_train)'-Prob_y_sachant_x*X_train';
Theta=Theta+taux_dapprentissage.*gradien

% Test validation

for k=1:4
    vector=zeros(1,4);
    vector(1,k)=1;
    P_Y_Validation(k,:)=exp(vector*Theta*X_validation);
end

Z_validation=sum(P_Y_Validation);
Prob_y_sachant_x_validation=P_Y_Validation;

for k=1:length(Z_validation)
    Prob_y_sachant_x_validation(:,k)=Prob_y_sachant_x_validation(:,k)/Z(1,k);
end

[value_find,class_find]=max(Prob_y_sachant_x_validation);
[value_real,class_real]=max(Y_validation');

Distance=sum((class_find-class_real)==0)


% Calculer la préc*ision sur l’ensemble d’apprentissage et
% l’ensemble de validation après chaque itération
% Utilisez l’ensemble de validation pour votre critère d’arrêt
% Calculer la mise à jour pour les paramètres

% Calculer la précision sur l’ensemble du test