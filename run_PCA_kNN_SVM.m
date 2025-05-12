% Author: John Connor Prikkel
% Date: 10/23/23
% ECE 595 Project 1

% Clear workspace
clear; close all; clc;

% Load the test file
d = load("Sample_MNIST.mat");

% X => 5000(M) x 400(n), Y => 5000 x 1
X = d.X;
y = d.y;

%Make sure to normalize training and validation data first!

% Get M (number of samples) and n (number of features)
[M, n] = size(X);

% Generate 100 random indicies ranging from 1 to 5000
%x_idx = randperm(5000, 100);

% Visualize these 100 cases of 'X'
%displayData(X(x_idx, :))

% Initialize k as num of classes
k = 10;

% Randomly pick 20% of cases for validation, select the images at these
% indices
%valid_idx = randperm(5000, 1000);
%valid_X = X(valid_idx, :);

% Get rest of indices for training
%train_idx = setdiff(1:5000, valid_idx);
%train_X = X(train_idx, :);

% Transpose labels
%valid_idx = valid_idx';
%train_idx = train_idx';

% Define number of folds
num_folds = 10;

% Use given number of features
N = 20;

% Use 'k_fold_indices' to create train_idx and test_idx for k-fold
% validation
[train_indices, test_indices] = k_fold_indices(M, num_folds);

% For each fold
for idx=1 : num_folds

    % Set the train and validation data to these indices
    train_X = X(train_indices{idx}, :);
    test_X = X(test_indices{idx}, :);

    % Initialize train and test labels
    train_labels = y(train_indices{idx})';
    test_labels = y(test_indices{idx})';

    % Perform PCA Transformation
    [A, Y_train, eigen_values] = PCA_transformation(train_X, N);

    % Multiply validation data w/ A
    Y_valid = test_X * A;

    % Apply fitcknn function (10 neighbors, squared inverse, euclidean
    % distance) -> default is euclidean
    kNN = fitcknn(Y_train, train_labels, 'NumNeighbors', 10, 'Distance', 'euclidean', 'DistanceWeight', 'squaredinverse', 'Standardize', 1);

    % Determine output of classifier
    predicted_labels (test_indices{idx}) = predict(kNN, Y_valid);

    % Create Gaussian + Quadratic templates for SVM classifiers
    Gaussian_template = templateSVM('KernelFunction', 'gaussian', 'PolynomialOrder', [], 'KernelScale', 6.3, 'BoxConstraint', 1, 'Standardize', 1);
    Quadratic_template = templateSVM('KernelFunction', 'polynomial','PolynomialOrder', 2, 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1);

    % Define multiclass classifiers
    Md1 = fitcecoc(Y_train, train_labels, 'Learners', Gaussian_template);
    Md2 = fitcecoc(Y_train, train_labels, 'Learners', Quadratic_template);
 
    % Make predictions using these classifiers
    predicted_g_labels (test_indices{idx}) = predict(Md1, Y_valid);
    predicted_q_labels (test_indices{idx}) = predict(Md2, Y_valid);
 
    % Transpose predicted_labels for ind2vec
    %predicted_g_labels = predicted_g_labels';
    %predicted_q_labels = predicted_q_labels';

end

% Plot Confusion Matrix
f1 = figure
kNNMatrix = plotconfusion(ind2vec(predicted_labels), ind2vec(y'));
 
% Plot Confusion Matricies
f2 = figure
gaussianMatrix = plotconfusion(ind2vec(predicted_g_labels), ind2vec(y'));

f3 = figure
quadraticMatrix = plotconfusion(ind2vec(predicted_q_labels), ind2vec(y'));

 





