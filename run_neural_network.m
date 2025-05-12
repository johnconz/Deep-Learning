% Author: John Connor Prikkel
% Date: 10/1/23
% ECE 595 Assignment 5

% Clear workspace
clear; close all; clc;

% Load the test file
d = load("Sample_MNIST.mat");

% X => 5000(M) x 400(n), Y => 5000 x 1
X = d.X;
y = d.y;

%Get M (number of samples) and n (number of features)
[M, n] = size(X);

% Generate 100 random indicies ranging from 1 to 5000
x_idx = randperm(5000, 100);

% Visualize these 100 cases of 'X'
displayData(X(x_idx, :))

% Initialize k as num of classes
k = 10;

% Initialize s2,s3 + thetas1-3 w/ random values btwn -0.12 and 0.12
s2 = 100;
s3 = 50;

%theta1 => s2 x (n+1)
%theta2 => s3 x (s2+1)
%theta3 => k x (s3+1)

% Scale to ensure between -0.12 and 0.12 since rand() generates btwn 0 and 1
theta1 = 0.12 + (-0.12 - (0.12)).*rand(s2, n + 1);
theta2 = 0.12 + (-0.12 - (0.12)).*rand(s3, s2 + 1);
theta3 = 0.12 + (-0.12 - (0.12)).*rand(k, s3 + 1);

% Insert ones into the first column of X
x0 = ones(M, 1);
X = [x0 X];

% Other knowns
alpha = 0.2;
num_iterations = 10000;

% NN Function with 2 hidden layers
[J, theta1, theta2, theta3] = nn_two_hidden_layers(theta1, theta2, theta3, X, y, num_iterations, alpha);

% Call determine_output fn
predicted_class = determine_output(theta1, theta2, theta3, X);

% Output final cost after iterations
fprintf("J(Error) = \n\n")
disp(J(end))

% find() returns the indicies where the actual feature's class matched the
% predicted, and taking the length() of this will return the number of respective indicies where this occured  
num_correct = length(find(y' == predicted_class));
accuracy = 100*(num_correct/M)
