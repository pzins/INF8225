% B    F
%  \ /
%   G
%  / \
% D   FT
% trait = fleche vers le bas
% bishop book chapter 8

B=1; F=2; G=3; D=4; FT=5;
names = cell(1,5);
names{B} = 'Battery'; names{F} = 'Fuel';
names{G} = 'Gauge';
names{D} = 'Distance'; names{FT} = 'Filltank';

dgm=zeros(5,5);
dgm(B,G)=1; 
dgm(G,D)=1;
dgm(F,G)=1; 
dgm(G,FT)=1;


% dgm(G,[G, FT])=1
CPDj{B} = tabularCpdCreate(reshape([0.1 0.9], 2, 1));
CPDj{F} = tabularCpdCreate(reshape([0.1 0.9], 2, 1));
CPDj{G} = tabularCpdCreate(reshape([0.9 0.8 0.8 0.2 0.1 0.2 0.2 0.8], 2, 2, 2))
CPDj{D} = tabularCpdCreate(reshape([0.95 0.7 0.05 0.03], 2, 2));


% P(G=1|B=1, F=1) = 0.8;
% P(G=0|B=1, F=1) = 0.2;
% 
% P(G=1|B=1, F=0) = 0.2;
% P(G=0|B=1, F=0) = 0.8;
% 
% P(G=1|B=0, F=1) = 0.2;
% P(G=0|B=0, F=1) = 0.8;
% 
% P(G=1|B=0, F=0) = 0.1;
% P(G=0|B=0, F=0) = 0.9;


% somme_sur_x(p(x|A,)) = 1
% somme_sur_x(p(x,y,z) = p(y,z)


% chaque ligne combinaison B et F : 00 01 10 11
% lignes : BF : 00 01  10 11
% P(G=0) P(G=1)
%  0.9    0.1
%  0.8    0.2

% P(G)=0 P(G)=1
%  0.8    0.2
%  0.2    0.8


% on def proba cond pr lesquelles G=0 (4 probas (les colonnes P(G=0)

% ensuite on fait pareil avc P(G=1)
% et on fait un reshape pr avoir un tensor
% 1 5
% 2 6
% 3 7
% 4 8
% reshape => tensor


% we compute P(F=0 | G=0)
% clamped = sparsevec(G,1,5);
% joint = dgmInferQuery(dgm, [B F G D FT]);
% pFgivenT = TabularFactorCondition(joint, F, clamped)
% fprintf('p(F=0|G=0=%f\n', pFgivenG.T(1));
% clamped = sparsevec(G,1,5);
% pFgivenCondB = TabularFactorCondition(joint, F, clamped);
% fprint('p(F=0|G=0, B=0 = %f\n', pfgivenGandB.T(1));
% result = P(F=0|G=0, B=0) = 0.11


% printed result = 0.257143


% on peut utiliser tabularFactorCondition(joint, X, clamped) qui utilise le
% force brute
% ou utiliser dgmInferQuery(dgm, X, 'clamped', clamped)
% normalement les resultats sont les mêmes


% sparsevec
% return vector with only at position G the value '2eme argument', 
% the rest is only 0
% 5 is the length of the vector

% remplissage des proba
% mettre combi des trucs sachant que (00 01 10 11) sur les lignes et mettre
% proba  aux colonnes P(X=0) P(X=1)
% ensuite pr remplir le tableau prendre premiere colonnes puis ajouté
% seconde colonne
