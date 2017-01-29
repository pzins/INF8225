fprintf('Question 1:\n');
fprintf(' [P]   [S]\n');
fprintf('   \\   /\n');
fprintf('    [C]\n');
fprintf('   /   \\\n');
fprintf(' [X]   [D]\n');
P = 1; S = 2; C = 3; X = 4; D = 5;

% noeuds du reseau bayesien
names = cell(1,5);
names{P} = 'PollutionHaute';
names{S} = 'Smoker';
names{C} = 'Cancer';
names{X} = 'XRay';
names{D} = 'Dispnea';

% dgm
dgm = zeros(5,5);
dgm([P,S],C) = 1;
dgm(C,[X,D]) = 1;

% probabilties
CPDj{P} = tabularCpdCreate(reshape([0.9 0.1], 2, 1));
CPDj{S} = tabularCpdCreate(reshape([0.7 0.3], 2, 1));
CPDj{C} = tabularCpdCreate(reshape([0.999 0.98 0.97 0.95 0.001 0.02 0.03 0.05],2 ,2 ,2));
CPDj{X} = tabularCpdCreate(reshape([0.8 0.1 0.2 0.9], 2,2));
CPDj{D} = tabularCpdCreate(reshape([0.7 0.35 0.3 0.65], 2, 2));

dgm = dgmCreate(dgm, CPDj, 'nodenames', names, 'infEngine', 'jtree');
joint = dgmInferQuery(dgm, [P S C X D]);


% Explaining away
fprintf('\n\n*******************************************************************************************************************************************************************************\n')
disp('Explaining Away');

% P(S=1|P=0)
clamped = sparsevec(P,1,5);
pS_sachant_nP = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|P=0) = %f\n', pS_sachant_nP.T(2))

% P(S=1|P=1)
clamped = sparsevec(P,2,5);
pS_sachant_P = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|P=1) = %f\n', pS_sachant_P.T(2))

% P(S=1)
pS = tabularFactorCondition(joint, S);
fprintf('P(S=1) = %f\n', pS.T(2));
disp('On remarque bien que P (pollution) et S (smoker) sont independants quand nous n''avons aucune information sur C (cancer)');

% Maintenant, supposons que l'on observe C
disp('Maintenant, supposons que l''on observe C')

% Dans ce cas, si l'on a une information sur P, la probabilité de S va
% changer
disp('Dans ce cas, si l''on a une information sur P, la probabilite de S va changer')
% si l'information est : P est vrai
disp('Si l''information est : P est vrai')
% P(S=1|C=1 P=1)
clamped = sparsevec([P C],[2],5);
pS_sachant_P_C = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|C=1 P=1) = %f\n', pS_sachant_P_C.T(2))
% P(S=1|C=1)
clamped = sparsevec([C],2,5);
pS_sachant_C = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|C=1) = %f\n', pS_sachant_C.T(2))
disp('On remarque bien que la probabilite P(S=1|C=1 P=1) a baissé par rapport à P(S=1|C=1) puisque le fait que C soit vrai est déjà expliqué par le fait que P soit vrai');

% De même si l'information est : P est faux
disp('De meme si l''information est : P est faux')
% P(S=1|C=1 P=0 )
clamped = sparsevec([P C],[1 2],5);
pS_sachant_C_nP = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|C=1 P=0) = %f\n', pS_sachant_C_nP.T(2))
% P(S=1|C=1)
clamped = sparsevec([C],2,5);
pS_sachant_C = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|C=1) = %f\n', pS_sachant_C.T(2))
disp('Cette fois, on remarque bien que la probabilite P(S=1|C=1 P=0) a augmenté par rapport à P(S=1|C=1) puisque le fait que C soit vrai devrait être expliqué par S puisque P est faux');
disp('*******************************************************************************************************************************************************************************')




fprintf('\n\n*******************************************************************************************************************************************************************************\n')
fprintf('Serial Block\n');
% Au depart, S, C et X sont dépendants : 
disp('Au depart, S, C et X sont dépendants :');
% P(X=1|S=0)
clamped = sparsevec(S,1,5);
pX_sachant_nS = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|S=0) = %f\n', pX_sachant_nS.T(2))
% P(X=1|S=1)
clamped = sparsevec(S,2,5);
pX_sachant_S = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|S=1) = %f\n', pX_sachant_S.T(2))
% P(X=1)
pX = tabularFactorCondition(joint, X);
fprintf('P(X=1) = %f\n', pX.T(2));
disp('Les trois probabilités ne sont pas égales, on remarque donc bien que S, C et X ne sont pas indépendants');

% Maintenant, C est observé et ainsi le phénomène serial blocking va rendre
% S et X indépendants
disp('% Maintenant, C est observé et ainsi le phénomène serial blocking va rendre S et X indépendants');
% P(X=1|C=1 S=0)
clamped = sparsevec([C S],[2 1],5);
X_sachant_C_nS = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|C=1 S=0) = %f\n', X_sachant_C_nS.T(2))
% P(X=1|C=1 S=1)
clamped = sparsevec([C S],[2 2],5);
X_sachant_C_S = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|C=1 S=1) = %f\n', X_sachant_C_S.T(2))
% P(X=1|C=1)
clamped = sparsevec([C],2,5);
X_sachant_C = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|C=1) = %f\n', X_sachant_C.T(2))
% Les 3 probabiltés sont égales, X et S sont donc bien indépendants
% conditionnellement à C
disp('Les 3 probabilités sont égales, X et S sont donc bien indépendants conditionnellement à C')
disp('*******************************************************************************************************************************************************************************')




fprintf('\n\n*******************************************************************************************************************************************************************************\n')
fprintf('Divergent blocking\n');
% Au depart, C, D et X sont dépendants : 
disp('Au depart, C, D et X sont dépendants :');
% P(X=1|D=0)
clamped = sparsevec(D,1,5);
pX_sachant_nD = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|D=0) = %f\n', pX_sachant_nD.T(2))
% P(X=1|D=1)
clamped = sparsevec(D,2,5);
pX_sachant_D = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|D=1) = %f\n', pX_sachant_D.T(2))
% P(X=1)
pX = tabularFactorCondition(joint, X);
fprintf('P(X=1) = %f\n', pX.T(2));
disp('Les trois probabilités ne sont pas égales, on remarque donc bien que C, D et X ne sont pas indépendants');

% Maintenant, C est observé et ainsi le phénomène divergent blocking va rendre
% D et X indépendants
disp('% Maintenant, C est observé et ainsi le phénomène divergent blocking va rendre D et X indépendants');

% P(X=1|C=1 D=0)
clamped = sparsevec([C D], [2 1], 5);
X_sachant_C_nD = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|C=1 D=0) = %f\n', X_sachant_C_nD.T(2));

% P(X=1|C=1 D=1)
clamped = sparsevec([C D], [2 2], 5);
X_sachant_C_D = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|C=1 D=1) = %f\n', X_sachant_C_D.T(2));

% P(X=1|C=1)
clamped = sparsevec([C], 2, 5);
X_sachant_C = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|C=1) = %f\n', X_sachant_C.T(2));
disp('Les 3 probabilités sont égales, X et D sont donc bien indépendants conditionnellement à C')
disp('*******************************************************************************************************************************************************************************')
