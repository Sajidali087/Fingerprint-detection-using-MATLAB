% Replace the input image directory with actual path of your dataset
query_image = imread('input dataset');
%replace Database image with you target directory, where image will be saved
database_dir = 'Database Image'; % Directory containing database images
database_files = dir(fullfile(database_dir, '*.jpg')); % Assuming all images are .png
query_grayscale = rgb2gray(query_image);
subplot(331); imshow(query_image); title('original image');
subplot(332); imshow(query_grayscale); title('gray scale image');
query_enhancedImage = histeq(query_grayscale);
query_clahe_image = adapthisteq(query_enhancedImage, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
subplot(333); imshow(query_clahe_image); title('enhanced image');
subplot(334); imhist(query_image);title('original image histogram');
subplot(335); imhist(query_clahe_image); title('enhanced image histogram');
query_bilateral_filtered = imbilatfilt(query_clahe_image, 0.2, 5);
subplot(336);imshow(query_bilateral_filtered); title('bilateral_filtered image');
se = ones(5);
level = graythresh(query_bilateral_filtered);
global_thresh = imbinarize(query_bilateral_filtered, level);

% Apply local (adaptive) thresholding
local_thresh = adaptthresh(query_bilateral_filtered, 0.4, 'ForegroundPolarity', 'dark', 'NeighborhoodSize', [11 11]);
binary_local = imbinarize(query_bilateral_filtered, local_thresh);
% Combine the results
query_combined_thresh = global_thresh & binary_local;
subplot(337); imshow(query_combined_thresh);
title('Combined Thresholding');

query_thinnedImage = bwmorph(query_combined_thresh, 'thin', Inf);
subplot(338); imshow(query_thinnedImage);
title('thinned Image');

% Skeletonize the image
query_skeleton_image = bwmorph(query_thinnedImage, 'skel', Inf);
% Minutiae extraction (example using simple method; for more accurate extraction, use specialized algorithms)
query_minutiae = bwmorph(query_skeleton_image, 'branchpoints'); % Find bifurcations
[ridge_end_y, ridge_end_x] = find(bwmorph(query_skeleton_image, 'endpoints')); % Find ridge endings
% Display the image with minutiae points
subplot(339); imshow(query_skeleton_image); hold on;
plot(ridge_end_x, ridge_end_y, 'ro'); % Ridge endings in red
[branch_y, branch_x] = find(query_minutiae);
plot(branch_x, branch_y, 'go'); % Bifurcations in green
hold off;
title('Minutiae Points on Skeletonized Fingerprint');


database_dir = 'C:\\Users\\OneDrive\\Desktop\\Resources';
database_files = dir(fullfile(database_dir, '*.jpg'));

for k = 1:length(database_files)
    % Construct the full file path correctly
    database_image = imread(fullfile(database_dir, database_files(k).name));
    
    % Preprocess the database image
    database_gray = rgb2gray(database_image);
    database_normalized = mat2gray(database_gray);
    database_enhanced = histeq(database_normalized);
    database_binary = imbinarize(database_enhanced);
    database_thinned = bwmorph(database_binary, 'thin', Inf);
    database_minutiae{k} = bwmorph(database_thinned, 'branchpoints') | bwmorph(database_thinned, 'endpoints');
end



% Initialize array to store match results
match_results = zeros(length(database_files), 1);

for k = 1:length(database_files)
    % Compare query image minutiae with current database image minutiae
    match_results(k) = matchMinutiae1(query_minutiae, database_minutiae{k});
end

% Find the best match
[~, best_match_index] = max(match_results);
disp(['Best match is: ', database_files(best_match_index).name, ' with ', num2str(match_results(best_match_index)), ' matching minutiae points']);


function match_count = matchMinutiae1(query_minutiae, db_minutiae)
    % Example matching threshold
    threshold = 10;
    match_count = 0;
    
    % Find the centroids of the minutiae points
    [query_y, query_x] = find(query_minutiae);
    [db_y, db_x] = find(db_minutiae);
    
    % Iterate over each minutiae point in the query image
    for i = 1:length(query_x)
        % Calculate distances to all minutiae points in the database image
        distances = sqrt((query_x(i) - db_x).^2 + (query_y(i) - db_y).^2);
        % Count matches within the threshold
        if any(distances < threshold)
            match_count = match_count + 1;
        end
    end
end