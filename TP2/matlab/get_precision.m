function [ precision ] = get_precision( X, Y, Theta)
%COMPUTEPRECISION Summary of this function goes here
%   Detailed explanation goes here


% compute de P(Y|X)
Z = sum(exp(Theta * X)); %denominator
P_Y_sachant_X = exp(Theta*X) ./ [Z;Z;Z;Z];


[myRes, myClass] = max(P_Y_sachant_X); %get the class with the highest probability
[realRes, realClass] = max(Y'); %real class
precision = sum(myClass==realClass)./length(myRes); %get the pourcentage of right answer


end

