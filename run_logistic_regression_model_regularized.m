% Author: John Connor Prikkel
% Date: 9/20/23
% ECE 595 Assignment 4

% Clear workspace
clear; close all; clc;

% Load the test file
d = load("Sample Data.txt");

% M = number of samples present, n = number of features
[M, n] = size(d);

% Feature vector equal to first 2 columns
x = d(:, 1:2);

% Desired value represented in 3rd column
y = d(:, 3);

%Indexes of cl
% find(X) returns a vector containing the indicies of ea nonzero ele in
% array X
class0_idx = find(~y); %all zero ele
class1_idx = find(y); %all nonzero ele

% DON'T NEED PLOTS FOR THIS ASSIGNMENT
% Plot distribution of the data 
%figure(1);
%plot(x(class0_idx, 1), x(class0_idx, 2), 'ro');
%hold on;
%plot(x(class1_idx, 1), x(class1_idx, 2), 'bo');
%title("Class Feature Distribution");
%xlabel("Feature x1");
%ylabel("Feature x2");
%legend({'Negative Class', 'Positive Class'}, 'Location', 'southwest');

% Normalize features
x_norm = normalize_features(x);

% Insert ones into the first column of x_norm
x0 = ones(M, 1);
x_norm = [x0 x_norm];

% Given Parameters
alpha = 0.1;
n_iterations = 2000;

% Test with different lambdas
for lambda = [0 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 1000]

    %initialize theta
    theta = [0,0,0];

    % Compute Cost
    J = compute_cost_logistic_regression_regularized(theta,x_norm,y,lambda);

    % Train using gradient descent 
    [J, theta] = gradient_descent_logistic_regression_regularized(theta,x_norm,y,alpha,n_iterations,lambda);

    % Print out lambda
    fprintf("lambda = \n\n")
    disp(lambda)

    % Print out theta on seperate line since don't want to print J
    fprintf("theta = \n\n")
    disp(theta)

    % DON'T NEED PLOTS FOR THIS ASSIGNMENT
    % Plot num_iterations against cost
    %figure(2);
    %plot(1:n_iterations, J(:));
    %title("Cost Function Plot");
    %xlabel("Number of Iterations");
    %ylabel("Cost");
    
    % Estimate y_hat by thresholding 'h'
    h = compute_sigmoid(theta*x_norm');
    y_hat = h > 0.5;
    
    % Turn y_hat (logical array) into a double array so it matches y's type
    y_hat = double(y_hat);
    
    % performance_measure() results
    [accuracy, C_matrix, precision, recall, f1, specificity] = performance_measure(y_hat, y)
end
