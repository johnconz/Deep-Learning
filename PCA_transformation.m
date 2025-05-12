% Author: John Connor Prikkel
% Date: 10/12/23
% ECE 595 Assignment 6

function [A,Y, eigen_values] = PCA_transformation(train_images, N)
% 'PCA_transformation' returns the PCA transformed
% features, transformation matrix and the corresponding Eigen values.
    
    % Determine the covariance matrix for each feature
    c = cov(train_images);

    % Determine the Eigen values (D) and Eigen vectors (V)
    % D is the diagonal matrix containing the eigen values 
    [V, D] = eig(c);

    
    % Sort eigen values in descending order; store indices they occur at
    [eigen_values, eigen_idx] = sort(diag(D), 'descend');

    % Change indices of Eigen vectors according to order of sorted Eigen
    % values as columns of V are eigenvectors
    V_updated = V(:, eigen_idx);
    
    % Choose Top N features, will be first N columns as sorted in
    % descending order
    A = V_updated(:, 1:N);
    %A = V_updated(:, 1)

    % (Also choose Top N eigenvalues - so that it is of the correct size N
    % x 1)
    eigen_values = eigen_values(1:N, :);
    %eigen_values = eigen_values(1, :);

    % Transform the given set of images
    Y = train_images * A;

end