
% unzip("DigitsData.zip")

imdsTrain = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\train", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsValidation = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\valid", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsTest = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\test", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% 
inputSize = [224 224 3];
numClasses = 10;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(4,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% 
options = trainingOptions('adam', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

%% 
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest)

%%
% imdsSingle = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\valid\Geonosis_Trippa_hive_Plains\696335481-geonosis3_2-078-jpg_jpg.rf.0a5dd7751bda58604aa9ebb72b47f9b9.jpg", ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% 
% YPred = classify(net, imdsSingle);
% YSingle = imdsSingle.Labels;
% accuracy = mean(YPred == YSingle)
%% 
% C= confusionmat(YPred, YTest);
cm = confusionchart(YPred,YTest);
cm.Title = 'Star Wars Battlefront 2 Map Classification with Image Classification Network';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

