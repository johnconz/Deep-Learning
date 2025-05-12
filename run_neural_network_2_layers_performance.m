% Author: John Connor Prikkel
% Date: 10/29/23
% ECE 595 Project 1

% Clear workspace
clear; close all; clc;

% Load the test file
d = load("malware_dataset.mat");

% data = 10868 x 4096 (images) double, label = 10868 x 1 double, filenm = 10868 x 20 char
data = d.data;
label = d.label;
fileName = d.filenm;

%Make sure to normalize data first!
data = normalize_features(data);

% Get M (number of samples) and n (number of features)
[M, n] = size(data);

% Insert ones into the first column of data
x0 = ones(M, 1);
data = [x0 data];

% 20% of dataset, or ~2174 images for validation
% Randomly pick 20% of cases for validation, select the images at these
% indices
valid_idx = randperm(10868, 2174);
valid_data = data(valid_idx, :);

% Get rest of indices for training, select images at these indices
train_idx = setdiff(1:10868, valid_idx);
train_data = data(train_idx, :);

% Initialize train and valid labels
train_labels = label(train_idx);
valid_labels = label(valid_idx);

% Initialize k as num of classes
k = 9;

% Vary s2,s3 (number of hidden units)
% Initialize thetas1-3 w/ random values btwn -0.12 and 0.12
s2 = 200;
s3 = 100;

%theta1 => s2 x (n+1)
%theta2 => s3 x (s2+1)
%theta3 => k x (s3+1)

% Scale to ensure between -0.12 and 0.12 since rand() generates btwn 0 and 1
theta1 = 0.12 + (-0.12 - (0.12)).*rand(s2, n + 1);
theta2 = 0.12 + (-0.12 - (0.12)).*rand(s3, s2 + 1);
theta3 = 0.12 + (-0.12 - (0.12)).*rand(k, s3 + 1);

% Test with different alphas (learning rates)
alpha = 0.2;

% Test with different number of iterations
num_iterations = 5000;

% NN Function with 2 hidden layers
[J, theta1, theta2, theta3] = nn_two_hidden_layers(theta1, theta2, theta3, train_data, train_labels, num_iterations, alpha);
        
% Call determine_output fn
predicted_class = determine_output(theta1, theta2, theta3, valid_data);
        
% Output final cost, alphas, amnt of iterations
fprintf("alpha = \n\n")
disp(alpha)

fprintf("J(Error) = \n\n")
disp(J(end))

% plot confusion matricies for validation data
f1 = figure
trainMatrix = plotconfusion(ind2vec(predicted_class), ind2vec(valid_labels'), "Validation Data");

