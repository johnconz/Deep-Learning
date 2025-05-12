% Author: John Connor Prikkel
% Date: 10/25/23
% ECE 595 Project 1

% Clear workspace
clear; close all; clc;

%Script to convert bytes files to images

% Using dir(), get all files with *.bytes extension in malware-classification\train folder
% Returned as a struct array
%train_bytesFiles = dir('D:\Connor\ECE 595\Project 1\malware-classification\train\*.bytes');

%pick one
train_bytesFiles = dir('D:\Connor\ECE 595\Project 1\malware-classification\train\0ACDbR5M3ZhBJajygTuf.bytes')

% Get number of .bytes files
N = size(train_bytesFiles);

% Read Excel file and note corresponding labels, use readtable() as
% readmatrix() requires homogenous variable types
trainLabelsTable = readtable('D:\Connor\ECE 595\Project 1\malware-classification\trainLabels.csv')

% Initialize array to store converted decimal codes
decimalArr = [];

%for each .bytes files in folder, where N is the number of .bytes files
for idx = 1:N

    % Change current directory so fopen() works (my .bytes files are in a
    % different folder than the MATLAB code)
    cd 'D:\Connor\ECE 595\Project 1\malware-classification\train'

    fileID = fopen(train_bytesFiles(idx).name);

    % Using textscan(), read data from open file into cell array
    % '%s' specifies the type as strings 17 times, as each line in file contains 17 strings
    fileCellArr = textscan(fileID, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s');

    % Save length of each cell as they can vary from file to file, but not
    % from cell to cell within a file
    cellLength = length(fileCellArr{2});

    % Can ignore first string as correponds to address, iterate over all
    % others
    for cell_idx = 2 : 17

        % For each pair of hex values within each cell
        for inner_cell_idx = 1 : cellLength

            % Can encounter the "??" character -> assign a corresponding
            % decimal value of -1
            if (fileCellArr{cell_idx}{inner_cell_idx} == '??')
               decimalData = -1; 
            else
                % Convert from hexadecimal to decimal (each set of 2 codes present
                % in cells 2 to 17)
                decimalData = hex2dec(fileCellArr{cell_idx}{inner_cell_idx});
            end

            % Concatenate new data to decimal code array
            % Concatenate vertically
            decimalArr = [decimalArr; decimalData];

        end
        
    
    end

    % Reshape decimalArr into a 2D array with same dimensions as .bytes file
    % Specify the length of the rows as 16(as from 2:17) and let MATLAB compute the number of
    % columns
    decimal2D = reshape(decimalArr, 16, [])

    % Return to folder that code is in
    cd 'D:\Connor\ECE 595\Project 1\MATLAB-Files\Project 1'

    % Convert each 2D array into images of size 256 x 16 using imagesc()
    f1 = figure;
    imageArray = imagesc(decimal2D)
    colormap("gray");

    % Excel files contain fileName and corresponding label, match fileID to
    % fileName in .csv file, described as 'ID's in table a
    label_idx = find(strcmp(fileID-6, trainLabelsTable.Id) == true)

    labels = trainLabelsTable.Class(label_idx)

    % Concantenate all images to matrix of size 10868 x 4096
    combImageMatrix = [combImageMatrix, imageArray]

    % Clear variables every loop iteration
    clearvars -except trainLabelsTable train_bytesFiles idx combImageMatrix labels

 end

% Save results (data and labels) in a .MAT file
save("malware_dataset.mat", "combImageMatrix", "labels")





