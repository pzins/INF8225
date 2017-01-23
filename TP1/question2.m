fprintf('Question 1:\n');
fprintf(' [C]   [T]\n');
fprintf('   \\   /\n');
fprintf('    [A]\n');
fprintf('   /   \\\n');
fprintf(' [M]   [J]\n');
C = 1; T = 2; A = 3; M = 4; J = 5;

% noeuds du reseau bayesien
names = cell(1,5);
names{C} = 'Cambriolage';
names{T} = 'Tremblement';
names{A} = 'Alarme';
names{M} = 'MarieAppelle';
names{J} = 'Jeanappelle';

% dgm
dgm = zeros(5,5);
dgm([C,T],A) = 1;
dgm(A,[M,J]) = 1;

% probabilties
CPDj{C} = tabularCpdCreate(reshape([0.999 0.001], 2, 1));
CPDj{T} = tabularCpdCreate(reshape([0.998 0.002], 2, 1));
CPDj{A} = tabularCpdCreate(reshape([0.999 0.71 0.60 0.05 0.001 0.29 0.94 0.95], 2, 2, 2));
CPDj{M} = tabularCpdCreate(reshape([0.95 0.1 0.05 0.9], 2,2));
CPDj{J} = tabularCpdCreate(reshape([0.99 0.3 0.01 0.7], 2, 2));

dgm = dgmCreate(dgm, CPDj, 'nodenames', names, 'infEngine', 'jtree');
joint = dgmInferQuery(dgm, [C T A M J]);

% affichage de l'histogramme de la probabilité jointe du réseau
lab = cellfun(@(x) {sprintf('%d ',x)}, num2cell(ind2subv([2 2 2 2],1:16),2));
figure;
bar(joint.T(:))
set(gca,'xtick',1:16);
xticklabelRot(lab, 90, 10, 0.01)
title('joint distribution of alarm')

% calculs
% P(C=1|M=1 J=0)
clamped = sparsevec([M J], [2 1], 5);
pC_sachant_M_nJ = tabularFactorCondition(joint, C, clamped);
fprintf('P(C=1|M=1 J=0) = %f\n', pC_sachant_M_nJ.T(2));

% P(C=1[M=0 J=1)
clamped = sparsevec([M J], [1 2], 5);
pC_sachant_nM_J = tabularFactorCondition(joint, C, clamped);
fprintf('P(C=1|M=0 J=1) = %f\n', pC_sachant_nM_J.T(2));

% P(C=1[M=1 J=1)
clamped = sparsevec([M J], [2 2], 5);
pC_sachant_M_J = tabularFactorCondition(joint, C, clamped);
fprintf('P(C=1|M=1 J=1) = %f\n', pC_sachant_M_J.T(2));

% P(C=1[M=0 J=0)
clamped = sparsevec([M J], [1 1], 5);
pC_sachant_nM_nJ = tabularFactorCondition(joint, C, clamped);
fprintf('P(C=1|M=0 J=0) = %f\n', pC_sachant_nM_nJ.T(2));

% P(C=1|M=1)
clamped = sparsevec([M], 2, 5);
pC_sachant_M = tabularFactorCondition(joint, C, clamped);
fprintf('P(C=1|M=1) = %f\n', pC_sachant_M.T(2));

% P(C=1|J=1)
clamped = sparsevec([J], 2, 5);
pC_sachant_J = tabularFactorCondition(joint, C, clamped);
fprintf('P(C=1|J=1) = %f\n', pC_sachant_J.T(2));

% Question d)
% P(C=1)
pC = tabularFactorCondition(joint, C);
fprintf('P(C=1) = %f\n', pC.T(2));
% P(T=1)
pT = tabularFactorCondition(joint, T);
fprintf('P(T=1) = %f\n', pT.T(2));
% P(A=1)
pA = tabularFactorCondition(joint, A);
fprintf('P(A=1) = %f\n', pA.T(2));
% P(M=1)
pM = tabularFactorCondition(joint, M);
fprintf('P(M=1) = %f\n', pM.T(2));
% P(J=1)
pJ = tabularFactorCondition(joint, J);
fprintf('P(J=1) = %f\n', pJ.T(2));