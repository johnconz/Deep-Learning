% Author: John Connor Prikkel
% Date: 10/1/23
% ECE 595 Assignment 5

function predicted_class = determine_output(theta1, theta2, theta3, X)
% predicted_class takes in parameters (𝜃1 , 𝜃2 and 𝜃3) along
% with input features and returns the output predicted labels

% a0 defined as bunch of ones, one for each of 2174 validation samples
a0 = ones(2174, 1);

% REPEAT SAME STEPS AS FORWARD PROPAGATION EXCEPT COST
%-- using formulae in slides--
z2 = X*theta1';
a2 = compute_sigmoid(z2);

% add bunch of ones
a2 = [a0 a2];

z3 = a2*theta2';
a3 = compute_sigmoid(z3);

% add bunch of ones
a3 = [a0 a3];

z4 = a3*theta3';
a4 = compute_sigmoid(z4);

% Transpose for use in max() fn
h = a4';

% determine the idx of max value value for each sample
[~,predicted_class]= max(h);

