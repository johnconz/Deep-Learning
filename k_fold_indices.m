% Author: John Connor Prikkel
% Date: 10/23/23
% ECE 595 Project 1

function [train_indices, test_indices] = k_fold_indices(num_images, num_folds)
% k_fold_indices takes in the number of images/datapoints and number of
% folds as input. It returns the train and test indices for each of the
% folds. Utilizing a cell array to save all train and test indices.

    % Generate random indicies equal to number of images ranging from 1 to
    % num_images, use to randomize test and train_indices
    r = randperm(num_images, num_images);    

    % Create n folds
    for idx = 1:num_folds
        
        % Choose every n indices to represent the test_indices (where n in
        % the value of num_folds)
        test_indices{idx} = r(idx : num_folds : num_images);

        % Define the train_indices as the remaining indices
        train_indices{idx} = setdiff(r, test_indices{idx});
          
    end

end