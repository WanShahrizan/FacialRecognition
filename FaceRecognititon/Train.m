% LOAD IMAGES(CHANGE FOLDER NAMES DEPENDING ON WHAT YOU SAVE)
outputFolder = fullfile('CNN');
rootFolder = fullfile(outputFolder, 'ObjectList');
%STORE IN DS
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
%SPLIT FOR TRAINING/TEST
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%LOAD NET
netAlex = alexnet;
%DISPLAY alexnet LAYERS
%analyzeNetwork(netAlex);

%DEFINE INPUT SIZE
inputSize = netAlex.Layers(1).InputSize;
%REPLACE FINAL LAYERS
layersTransfer = netAlex.Layers(1:end-3);
%CLASSES DEPENDS ON HOW MANY LABELS WE HAVE
numClasses = numel(categories(imdsTrain.Labels));
numClasses = 3;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%DATA AUGMENTATION
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandXTranslation',...
    pixelRange,'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%TRAIN NETWORK
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

%CLASSIFY TEST IMAGE USING netTransfer
[YPred,scores] = classify(netTransfer,augimdsValidation);

%DISPLAY CLASSIFIED IMAGE & THEIR LABELS
idx = randperm(numel(imdsValidation.Files),3);
figure
for i = 1:3
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%PREDICT ACCURACY
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
