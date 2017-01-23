fprintf('Question 1:\n');
fprintf(' [P]   [S]\n');
fprintf('   \\   /\n');
fprintf('    [C]\n');
fprintf('   /   \\\n');
fprintf(' [X]   [D]\n');
P = 1; S = 2; C = 3; X = 4; D = 5;

names = cell(1,5);
names{P} = 'PollutionHaute';
names{S} = 'Smoker';
names{C} = 'Cancer';
names{X} = 'XRay';
names{D} = 'Dispnea';

dgm = zeros(5,5);
dgm([P,S],C) = 1;
dgm(C,[X,D]) = 1;

CPDj{P} = tabularCpdCreate(reshape([0.9 0.1], 2, 1));
CPDj{S} = tabularCpdCreate(reshape([0.7 0.3], 2, 1));
CPDj{C} = tabularCpdCreate(reshape([0.001 0.02 0.03 0.05 0.999 0.98 0.97 0.95], 2, 2, 2));
CPDj{X} = tabularCpdCreate(reshape([0.8 0.1 0.2 0.9], 2,2));
CPDj{D} = tabularCpdCreate(reshape([0.7 0.35 0.3 0.65], 2, 2));

dgm = dgmCreate(dgm, CPDj, 'nodenames', names, 'infEngine', 'jtree');
joint = dgmInferQuery(dgm, [P S C X D]);

% sparsevec
% return vector with only at position G the value '2eme argument', 
% the rest is only 0
% 5 is the length of the vector

% Explaining away
disp('Explaining Away');

% P(S=1|P=1)
clamped = sparsevec(P,2,5);
S_sachant_P = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|P=1) = %f\n', S_sachant_P.T(2))

% P(S=1|P=0)
clamped = sparsevec(P,1,5);
S_sachant_P = tabularFactorCondition(joint, S, clamped);
fprintf('P(S=1|P=0) = %f\n', S_sachant_P.T(2))
disp('De plus P(S=1)=0.3')
disp('On remarque bien que P (pollution) et S (smoker) sont independants quand nous avons aucune information sur C (cancer)');

% Maintenant avec une information sur C
disp('Maintenant avec une information sur C')

% P(S=1|P=0 C=0)
clamped = sparsevec([P C],1,5);
S_sachant_P_C = tabularFactorCondition(joint, X, clamped);
fprintf('P(S=1|P=1 C=1) = %f\n', S_sachant_P_C.T(1))
disp('En ayant une information sur C (Cancer, on remarque que si on a un indice')
disp('sur P (Pollution) qui se realise alors S (smoker) voit sa probabilité chuter')

fprintf('\n\n Serial Block\n');
% P(X=1|S=0 C=1)
clamped = sparsevec([S C],[1 2],5);
X_sachant_S_C = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|S=0 C=1) = %f\n', X_sachant_S_C.T(1))
% P(X=1|S=1 C=1)
clamped = sparsevec([S C],[2 2],5);
X_sachant_S_C = tabularFactorCondition(joint, X, clamped);
fprintf('P(X=1|S=1 C=1) = %f\n', X_sachant_S_C.T(1))
disp('On remarque donc que une information sur C (Cancer) rend S (smoker) et X (Xray) independants')


