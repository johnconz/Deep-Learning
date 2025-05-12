% Author: John Connor Prikkel
% Date: 10/1/23
% ECE 595 Assignment 5

function predicted_class = determine_output_three_layers(theta1, theta2, theta3, theta4, X)
% predicted_class takes in parameters (𝜃1 , 𝜃2, 𝜃3, and 𝜃4) along
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

% add bunch of ones
a4 = [a0 a4];

z5 = a4*theta4';
a5 = compute_sigmoid(z5);

% Transpose for use in max() fn
h = a5';

% determine the idx of max value value for each sample
[~,predicted_class]= max(h);

