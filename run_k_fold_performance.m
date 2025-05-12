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

% Define number of folds
num_folds = 10;

% Use variable number of features
N = 65;

% Use 'k_fold_indices' to create train_idx and test_idx for k-fold
% validation
[train_indices, test_indices] = k_fold_indices(M, num_folds);

% For each fold
for idx=1 : num_folds

    % Set the train and validation data to these indices
    train_data = data(train_indices{idx}, :);
    test_data = data(test_indices{idx}, :);

    % Initialize train and test labels
    train_labels = label(train_indices{idx})';
    test_labels = label(test_indices{idx})';

    % Perform PCA Transformation
    [A, Y_train, eigen_values] = PCA_transformation(train_data, N);

    % Multiply validation data w/ A
    Y_valid = test_data * A;

    % Apply fitcknn function (10 neighbors, squared inverse, euclidean
    % distance) -> default is euclidean
    kNN_euclidean = fitcknn(Y_train, train_labels, 'NumNeighbors', 10, 'Distance', 'euclidean', 'DistanceWeight', 'squaredinverse', 'Standardize', 1);
    
    % Apply other fitcknn functions
    % Minkowski - default exponent is 2, use 1 for manhattan
    % Chebychev - Maximum Coordinate Difference
    kNN_minkowski = fitcknn(Y_train, train_labels, 'NumNeighbors', 10, 'Distance', 'minkowski', 'Exponent', 1, 'DistanceWeight', 'squaredinverse', 'Standardize', 1);
    kNN_chebychev = fitcknn(Y_train, train_labels, 'NumNeighbors', 10, 'Distance', 'chebychev', 'DistanceWeight', 'squaredinverse', 'Standardize', 1);

    % Determine output of classifier
    predicted_e_labels (test_indices{idx}) = predict(kNN_euclidean, Y_valid);
    predicted_m_labels (test_indices{idx}) = predict(kNN_minkowski, Y_valid);
    predicted_ch_labels (test_indices{idx}) = predict(kNN_chebychev, Y_valid);

    % Create templates for SVM classifiers
    Gaussian_template = templateSVM('KernelFunction', 'gaussian', 'PolynomialOrder', [], 'KernelScale', 6.3, 'BoxConstraint', 1, 'Standardize', 1);
    Quadratic_template = templateSVM('KernelFunction', 'polynomial','PolynomialOrder', 2, 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1);
    Cubic_template = templateSVM('KernelFunction', 'polynomial','PolynomialOrder', 3, 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1);

    % Define multiclass classifiers
    Md1 = fitcecoc(Y_train, train_labels, 'Learners', Gaussian_template);
    Md2 = fitcecoc(Y_train, train_labels, 'Learners', Quadratic_template);
    Md3 = fitcecoc(Y_train, train_labels, 'Learners', Cubic_template);
 
    % Make predictions using these classifiers
    predicted_g_labels (test_indices{idx}) = predict(Md1, Y_valid);
    predicted_q_labels (test_indices{idx}) = predict(Md2, Y_valid);
    predicted_c_labels (test_indices{idx}) = predict(Md3, Y_valid);

end

% Plot Confusion Matricies
f1 = figure
kNNEuclideanMatrix = plotconfusion(ind2vec(predicted_e_labels), ind2vec(label'), "kNN (Euclidean)");

f2 = figure
kNNMinkowskiMatrix = plotconfusion(ind2vec(predicted_m_labels), ind2vec(label'), "kNN (Minkowski)");

f3 = figure
kNNChebychevMatrix = plotconfusion(ind2vec(predicted_ch_labels), ind2vec(label'), "kNN (Chebychev)");
 
f4 = figure
gaussianMatrix = plotconfusion(ind2vec(predicted_g_labels), ind2vec(label'), "SVM (Gaussian)");

f5 = figure
quadraticMatrix = plotconfusion(ind2vec(predicted_q_labels), ind2vec(label'), "SVM (Quadratic)");

f6 = figure
cubicMatrix = plotconfusion(ind2vec(predicted_c_labels), ind2vec(label'), "SVM (Cubic)");

 




