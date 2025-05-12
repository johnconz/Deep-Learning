

function I_adapthisteq = preprocess_malaria_images(filename, desired_size)

% This function preprocesses malaria images using contrast enhancement and
% resizes the image
% Author: Connor Prikkel
% Ref: https://www.mathworks.com/help/images/contrast-enhancement-techniques.html

% Read the Image
I = imread(filename);

% Some images might be grayscale, replicate the image 3 times to
% create an RGB image.

if ismatrix(I)
  I = cat(3,I,I,I);
end

% Convert the image from RGB color space to L*a*b* color space
I_lab = rgb2lab(I);

% Values of luminosity span a range from 0 to 100 
% Scale values to range [0 1]
max_luminosity = 75;
L = I_lab(:,:,1)/max_luminosity;


% Contrast-Limited Adaptive Histogram Equalization - operates on small
% sections of the intensity image at a time
I_adapthisteq = I_lab;
I_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
I_adapthisteq = lab2rgb(I_adapthisteq);

% Resize the image
I_adapthisteq = imresize(I_adapthisteq, [desired_size(1) desired_size(2)]);

end

