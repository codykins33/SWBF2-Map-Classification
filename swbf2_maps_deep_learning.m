
trainingSet = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\train", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsValidation = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\valid", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

testSet = imageDatastore("C:\Users\codyl\Downloads\swbf2_maps\test", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


%% 
% Load pretrained network
net = resnet50();
%% 
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
%% 
% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')
%% 
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% 
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
%% 
% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))
%% 
[predictedLabels, predictedScores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

%% 
cm = confusionchart(testLabels, predictedLabels)
cm.Title = 'Star Wars Battlefront 2 Map Classification with Deep Learning Network';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
%% 
% Visualize the first section of the network. 
figure
plot(net)
title('First section of ResNet-50')
set(gca,'YLim',[150 170]);
%% 
chosenClass = "Geonosis_Trippa_hive_Plains";
classIdx = find(net.Layers(end).Classes == chosenClass);

numImgsToShow = 4;

[sortedScores,imgIdx] = findMaxActivatingImages(testSet,chosenClass,predictedScores,numImgsToShow);

% figure
% plotImages(testSet,imgIdx,sortedScores,predictedLabels,numImgsToShow)
%% 

imageNumber = 1;

observation = augmentedTestSet.readByIndex(imgIdx(imageNumber));
img = observation.input{1};

label = predictedLabels(imgIdx(imageNumber));
score = sortedScores(imageNumber);

gradcamMap = gradCAM(net,img,label);

figure
alpha = 0.5;
plotGradCAM(img,gradcamMap,alpha);
sgtitle(string(label)+" (score: "+ max(score)+")")

%% 
img = imread("C:\Users\codyl\Downloads\swbf2_maps\train\Geonosis_Trippa_hive_Plains\956237735-geonosis3_2-019-jpg_jpg.rf.e1a0f508c625a256a9cce02689df40cb.jpg");
img = imresize(img,net.Layers(1).InputSize(1:2),"Method","bilinear","AntiAliasing",true);

[label,score] = classify(net,img);

gradcamMap = gradCAM(net,img,label);
 

figure
alpha = 0.5;
plotGradCAM(img,gradcamMap,alpha);
title(string(label)+" (score: "+ max(score)+")")

%% 
img = imread("C:\Users\codyl\Downloads\swbf2_maps\train\Kamino_Cloning_facility_Archives\410711529-kamino1-1-06-jpg_jpg.rf.b52f3ec51fde3c87fa96c740be9f80ef.jpg");
img = imresize(img,net.Layers(1).InputSize(1:2),"Method","bilinear","AntiAliasing",true);

[label,score] = classify(net,img);

gradcamMap = gradCAM(net,img,label);

figure
alpha = 0.5;
plotGradCAM(img,gradcamMap,alpha);
title(string(label)+" (score: "+ max(score)+")")

%% 
img = imread("C:\Users\codyl\Downloads\swbf2_maps\train\Naboo_Theed_Palace_control_room\966925997-naboo_3_2-09-jpg_jpg.rf.dd621f571d4d0f7708d92b428aa05ec6.jpg");
img = imresize(img,net.Layers(1).InputSize(1:2),"Method","bilinear","AntiAliasing",true);

[label,score] = classify(net,img);

gradcamMap = gradCAM(net,img,label);

figure
alpha = 0.5;
plotGradCAM(img,gradcamMap,alpha);
title(string(label)+" (score: "+ max(score)+")")

%% 
function [sortedScores,imgIdx] = findMaxActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in descending order
[sortedScores,idx] = sort(scoresForChosenClass,'descend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end

function plotGradCAM(img,gradcamMap,alpha)

subplot(1,2,1)
imshow(img);

h = subplot(1,2,2);
imshow(img)
hold on;
imagesc(gradcamMap,'AlphaData',alpha);

originalSize2 = get(h,'Position');

colormap jet
colorbar

set(h,'Position',originalSize2);
hold off;
end

function [scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores)
% Find the index of className (e.g. "sushi" is the 9th class)
uniqueClasses = unique(imds.Labels);
chosenClassIdx = string(uniqueClasses) == className;

% Find the indices in imageDatastore that are images of label "className"
% (e.g. find all images of class sushi)
imgsOfClassIdxs = find(imds.Labels == className);

% Find the predicted scores of the chosen class on all the images of the
% chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
scoresForChosenClass = predictedScores(imgsOfClassIdxs,chosenClassIdx);
end


