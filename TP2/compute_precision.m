function [ precision ] = compute_precision( X, Y, Theta)
%COMPUTEPRECISION Summary of this function goes here
%   Detailed explanation goes here


% calcul de P(Y|X)
up = exp(Theta*X); %numerateur
Z = sum(exp(Theta * X)); %denominateur
P_Y_sachant_X = up ./ [Z;Z;Z;Z];

myRes = max(P_Y_sachant_X); %get the probability of the most probable class
realRes = full(sum(Y' .* P_Y_sachant_X)); %get probability of the real class
precision = sum(myRes==realRes)./length(myRes); %get the pourcentage of right answer


end

