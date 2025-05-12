% Author: John Connor Prikkel
% Date: 10/23/23
% ECE 595 Project 1

function [J, theta1, theta2, theta3] = nn_two_hidden_layers(theta1, theta2, theta3, X, y, num_iterations, alpha)
% nn_n_hidden_layers returns the cost and updated weight values
% after the backpropagation step

    % need to change values of y to vectorized format
    y = ind2vec(y');

    m = length(y);

    % a0 defined as bunch of ones, one for all 8694 train samples
    a0 = ones(8694, 1);

    for idx=1:num_iterations

        % FORWARD PROPAGATION
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
      
        % Compute Cost, a4 = h
        J(idx) = -1/m*sum(sum((y'.*log(a4)+(1-y').*log(1-a4))));

        % BACKWARD PROPAGATION
        %-- using formulae in slides--

        delta4 = a4 - y';
        delta3 = (delta4*theta3).*(a3.*(1-a3));

        %remove first column of matrix
        delta3(:, 1) = [];

        delta2 = (delta3*theta2).*(a2.*(1-a2));

        %remove first column of matrix
        delta2(:, 1) = [];

        %update thetas
        theta3 = theta3 - ((alpha.*(delta4)'*a3)/m);
        theta2 = theta2 - ((alpha.*(delta3)'*a2)/m);
        theta1 = theta1 - ((alpha.*(delta2)'*X)/m);
    end

