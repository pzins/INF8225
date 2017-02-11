function [ precision ] = get_precision( X, Y, Theta)
%COMPUTEPRECISION Summary of this function goes here
%   Detailed explanation goes here


% calcul de P(Y|X)
up = exp(Theta*X); %numerateur
Z = sum(exp(Theta * X)); %denominateur
P_Y_sachant_X =  up ./ [Z;Z;Z;Z];


[myRes, myClass] = max(P_Y_sachant_X); %get the class with the highest probability
[realRes, realClass] = max(Y'); %real class
precision = sum(myClass==realClass)./length(myRes); %get the pourcentage of right answer


end

