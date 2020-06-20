rootFolder = 'Celebrity';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

labels = classify(net, imds_test);
ii = randi(4000);
im = imread(img_1{ii});
imshow(im);
