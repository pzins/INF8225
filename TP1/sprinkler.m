C = 1; S = 2; R = 3; W = 4;
nvars = 4;
dgm = mkSprinklerDgm();
% if ~isOctave
%     drawNetwork('-adjMatrix', dgm.G, '-nodeLabels', {'C', 'S', 'R', 'W'},...
%         '-layout', Treelayout);
% end

joint = dgmInferQuery(dgm, [C S R W])
% T : proba
% domain et sizes

% ind2subv : convertit les éléments 1:16 sur pls dimension (ici 4)
% avec des indices 1 ou 2
% num2cell : convetit le tableau en tableau de cellule
% cellfun : applique la fct (1er argument) à chaque cellule du tableau
% (2eme argument)
lab = cellfun(@(x) {sprintf('%d ', x)}, num2cell(ind2subv([2 2 2 2],1:16),2));

% figure;
% bar(joint.T(:))
% set(gca,'xtick',1:16);
% xticklabelRot(lab, 90, 10, 0.01)
% title('joint distribution of water sprinkler UGM')

% manuellement
fac{C} = tabularFactorCreate(reshape([0.5 0.5], 2, 1), [C]);
fac{S} = tabularFactorCreate(reshape([0.5 0.9 0.5 0.1], 2, 2), [C S]);
fac{R} = tabularFactorCreate(reshape([0.8 0.2 0.2 0.8], 2, 2), [C R]);
fac{W} = tabularFactorCreate(reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], 2, 2, 2), [S R W]);
jointF = tabularFactorMultiply(fac);
assert(tfequal(joint, jointF));

pW = dgmInferQuery(dgm, W)
pS = dgmInferQuery(dgm, S)
pS.T