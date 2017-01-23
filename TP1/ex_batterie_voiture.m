fprintf('TP1 - Question 1:\n\n');

fprintf(' [B]   [F]\n');
fprintf('   \\   /\n');
fprintf('    [G]\n');
fprintf('   /   \\\n');
fprintf(' [D]   [FT]\n\n');

B = 1; F = 2; G = 3; D = 4; FT = 5;

names = cell(1,5);
names{B}  = 'Battery';
names{F}  = 'Fuel';
names{G}  = 'Gauge';
names{D}  = 'Distance';
names{FT} = 'BatteryFillTank';

dgm = zeros(5,5);
dgm(B,G) = 1;
dgm(F,G) = 1;
dgm(G,[D, FT]) = 1;

CPDs{B} = tabularCpdCreate(reshape([0.1 0.9], 2, 1));
CPDs{F} = tabularCpdCreate(reshape([0.1 0.9], 2, 1));
CPDs{G} = tabularCpdCreate(reshape([0.9 0.8 0.8 0.2 0.1 0.2 0.2 0.8], 2, 2, 2));
CPDs{D} = tabularCpdCreate(reshape([0.95 0.7 0.05 0.3], 2, 2));
CPDs{FT} = tabularCpdCreate(reshape([0.2 0.6 0.8 0.4], 2, 2));

dgm = dgmCreate(dgm, CPDs, 'nodenames', names, 'infEngine', 'jtree');
joint = dgmInferQuery(dgm, [B,F,G,D,FT]);

fprintf('(1): Explaining Away\n');
clampled = sparsevec(B,1,5);
F_Sachant_B = tabularFactorCondition(joint, F, clampled);
fprintf('F Sachant !B: p(F|B=0)=%f\n', F_Sachant_B.T(1));

clampled = sparsevec(B,2,5);
F_Sachant_B = tabularFactorCondition(joint, F, clampled);
fprintf('F Sachant  B: p(F|B=1)=%f\n', F_Sachant_B.T(1));
fprintf('Il n''y a aucuns changements pour F sachant B car B et F ne sont pas dependants\n\n');

clampled = sparsevec([B G],2,5);
F_Sachant_B_G = tabularFactorCondition(joint, F, clampled);
fprintf('F Sachant  B  G): p(F|B=1, G=1)=%f\n', F_Sachant_B_G.T(1));
fprintf('En fixant G _et_ B, on peut voir la probabilite de F augmenter par un ''Explaining Away''\n\n\n');
break

fprintf('(2): Serial Blocking\n');
clampled = sparsevec(B, 1, 5);
D_Sachant_B = tabularFactorCondition(joint, D, clampled);
fprintf('D Sachant !B: p(D|B=0)=%f\n', D_Sachant_B.T(1));

clampled = sparsevec(B, 2, 5);
D_Sachant_B = tabularFactorCondition(joint, D, clampled);
fprintf('D Sachant  B: p(D|B=1)=%f\n', D_Sachant_B.T(1));
fprintf('On voit que B influence D\n\n');

clampled = sparsevec([B G], 1, 5);
D_Sachant_B_G = tabularFactorCondition(joint, D, clampled);
fprintf('D Sachant !B !G: p(D|B=0, G=0)=%f\n', D_Sachant_B_G.T(1));

clampled = sparsevec([B G], [2 1], 5);
D_Sachant_B_G = tabularFactorCondition(joint, D, clampled);
fprintf('D Sachant  B !G: p(D|B=1, G=0)=%f\n', D_Sachant_B_G.T(1));
fprintf('On voit que G bloque l''influence de B sur G\n\n\n');


fprintf('(3): Divergent Blocking\n');
clampled = sparsevec(D,1,5);
FT_Sachant_D = tabularFactorCondition(joint, FT, clampled);
fprintf('FT Sachant !D: p(FT|D=0)=%f\n',FT_Sachant_D.T(1));

clampled = sparsevec(D,2,5);
FT_Sachant_D = tabularFactorCondition(joint, FT, clampled);
fprintf('FT Sachant  D: p(FT|D=1)=%f\n',FT_Sachant_D.T(1));
fprintf('On voit que D influence FT\n\n');

clampled = sparsevec([D G],1,5);
FT_Sachant_D_G = tabularFactorCondition(joint, FT, clampled);
fprintf('FT Sachant  D  G: p(FT|D=0, G=0)=%f\n',FT_Sachant_D_G.T(1));

clampled = sparsevec([D G],[2 1],5);
FT_Sachant_D_G = tabularFactorCondition(joint, FT, clampled);
fprintf('FT Sachant  D !G: p(FT|D=1, G=0)=%f\n',FT_Sachant_D_G.T(1));
fprintf('On voit que G bloque l''influence de D sur FT\n');