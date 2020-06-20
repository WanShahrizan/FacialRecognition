%%Extracting Image
categories = {'Scarlett','Robert','Mark','Chris'};

rootFolder = 'Celebrity';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
%%Resize Image Missing

%Define Layer
varSize = 227;
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2,'WeightLearnRateFactor',5);
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2,'WeightLearnRateFactor',5);
fc2 = fullyConnectedLayer(4,'BiasLearnRateFactor',2,'WeightLearnRateFactor',5);

layers = [
    imageInputLayer([varSize varSize 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];

%%Training Option
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'Verbose', true, ...
    'Plots','training-progress');

[net, info] = trainNetwork(imds, layers, opts);
