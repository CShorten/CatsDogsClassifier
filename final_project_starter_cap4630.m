%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Fall 2016 
%% Final Project - Starter code (2-class classifier: cats vs. dogs)

% Inspired by the example "Deep Learning for Pet Classification" 
% (Copyright 2016 The MathWorks, Inc.)

%% Part 1: Download, load and inspect Pre-trained Convolutional Neural Network (CNN)
% You will need to download a pre-trained CNN model for this example.
% There are several pre-trained networks that have gained popularity.
% Most of these have been trained on the ImageNet dataset, which has 1000
% object categories and 1.2 million training images[1]. "AlexNet" is one
% such model and can be downloaded from MatConvNet[2,3]. 

%% 1.1: Location of pre-trained "AlexNet"
%cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
% Specify folder for storing CNN model 
cnnFolder = './networks'; %Modified this from ./networks
cnnMatFile = 'imagenet-caffe-alex.mat';
cnnFullMatFile = fullfile(cnnFolder, cnnMatFile);

%Check that the code is only downloaded once
%if ~exist(cnnFullMatFile, 'file')
%    disp('Downloading pre-trained CNN model...');     
%    websave(cnnFullMatFile, cnnURL);
%end




%% 1.2: Load Pre-trained CNN
% The CNN model is saved in MatConvNet's format [3]. Load the MatConvNet
% network data into |convnet|, a |SeriesNetwork| object from Neural Network
% Toolbox(TM), using the helper function |helperImportMatConvNet| in the 
% Computer Vision System Toolbox (TM). A SeriesNetwork object can be used
% to inspect the network architecture, classify new data, and extract
% network activations from specific layers.

%MatConvNet network into a SeriesNetwork
convnet = helperImportMatConvNet(cnnFullMatFile);

%% 1.3: Inspsect pre-trained CNN
% |convnet.Layers| defines the architecture of the CNN 
% Inspect the name, size, and properties of the CNN's layers
convnet.Layers

% The intermediate layers make up the bulk of the CNN. These are a series
% of convolutional layers, interspersed with rectified linear units (ReLU)
% and max-pooling layers [2]. Following the these layers are 3
% fully-connected layers.
%
% The final layer is the classification layer and its properties depend on
% the classification task. In this example, the CNN model that was loaded
% was trained to solve a 1000-way classification problem. Thus the
% classification layer has 1000 classes from the ImageNet dataset. 

% Inspect the last layer
disp('Inspect the last layer')
convnet.Layers(end)

% Number of class names for ImageNet classification task
numel(convnet.Layers(end).ClassNames)

%%
% Note that the CNN model is not going to be used for the original
% classification task. It is going to be re-purposed to solve a different
% classification task on the pets dataset.

%% Part 2: Set up image d ata
%% 2.1: Load simplified dataset and build image store
dataFolder = './data/PetImages';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds) 

% Use the smallest overlap set
% (useful when the two classes have different number of elements)
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%% 2.2: Pre-process Images For CNN
% |convnet| can only process RGB images that are 227-by-227.
% To avoid re-saving all the images to this format, setup the |imds|
% read function, |imds.ReadFcn|, to pre-process images on-the-fly.
% The |imds.ReadFcn| is called every time an image is read from the
% |ImageDatastore|.
%
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% 2.3: Divide data into training and testing sets
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');


%% Part 3: Feature Extraction 
% Extract training features using pretrained CNN

% Each layer of a CNN produces a response, or activation, to an input
% image. However, there are only a few layers within a CNN that are
% suitable for image feature extraction. The layers at the beginning of the
% network capture basic image features, such as edges and blobs. To see
% this, visualize the network filter weights from the first convolutional
% layer. This can help build up an intuition as to why the features
% extracted from CNNs work so well for image recognition tasks. Note that
% visualizing deeper layer weights is beyond the scope of this example. You
% can read more about that in the work of Zeiler and Fergus [4].

%% 3.1: Inspect the network weights for the second convolutional layer
% Get the network weights for the second convolutional layer
w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. 
figure
montage(w1)
title('First convolutional layer weights')

%%% Visualize Second Layer
channels = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16];
%w2 = deepDreamImage(convnet,6,channels);
%figure
%montage(w2)
%title('Second convolutional layer weights w/ deepDreamImage')

%%% Visualize Third Layer
%w3 = deepDreamImage(convnet,10,channels);
%figure
%montage(w3)
%title('Third Convolutional Layer weights w/ deepDreamImage')

%%% Visualize FourthLayer
%w4 = deepDreamImage(convnet,12,channels);
%figure
%montage(w4)
%title('Fourth Convolutional Layer weights w/ deepDreamImage')

% Notice how the first layer of the network has learned filters for
% capturing blob and edge features. These "primitive" features are then
% processed by deeper network layers, which combine the early features to
% form higher level image features. These higher level features are better
% suited for recognition tasks because they combine all the primitive
% features into a richer image representation [5].

%% 3.2: Use features from one of the deeper layers
% You can easily extract features from one of the deeper layers using the
% |activations| method. Selecting which of the deep layers to choose is a
% design choice, but typically starting with the layer right before the
% classification layer is a good place to start. In |convnet|, the this
% layer is named 'fc7'. Let's extract training features using that layer.

featureLayer = 'fc7';

trainingFeaturesFolder = './';
trainingFeaturesFile = 'trainingFeatures.mat'; 
trainingFeaturesFullMatFile = fullfile(trainingFeaturesFolder, trainingFeaturesFile);

% Check that the code is only downloaded once
if ~exist(trainingFeaturesFullMatFile, 'file')
    disp('Building training features... This will take a while...');     
    trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
      'MiniBatchSize', 32, 'OutputAs', 'columns');
    save(trainingFeaturesFullMatFile, 'trainingFeatures');
else
    load trainingFeatures.mat
end

% Note that the activations are computed on the GPU and the 'MiniBatchSize'
% is set 32 to ensure that the CNN and image data fit into GPU memory.
% You may need to lower the 'MiniBatchSize' if your GPU runs out of memory.
%
% Also, the activations output is arranged as columns. This helps speed-up
% the multiclass linear SVM training that follows.

%% Part 4: Train a multiclass SVM classifier
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
alternateSVMClassifier = fitclinear(trainingFeatures, trainingLabels,...
    'ObservationsIn', 'columns', 'Learner', 'svm');
TreeClassifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'tree', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
NaiveBayesClassifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'naivebayes', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Part 5: Evaluate classifier

%% 5.1: Extract features from images in the test set
testFeaturesFolder = './';
testFeaturesFile = 'testFeatures.mat'; 
testFeaturesFullMatFile = fullfile(testFeaturesFolder, testFeaturesFile);

% Check that the code is only downloaded once
if ~exist(testFeaturesFullMatFile, 'file')
    disp('Extracting features... This will take a while...');     
    % Extract test features using the CNN
    testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);
    % Save features for future use
    save(testFeaturesFullMatFile, 'testFeatures');
else
    load testFeatures.mat
end

%% 5.2: Test classifier's prediction accuracy and produce confusion matrix
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);
%predictedLabels2 = predict(alternateSVMClassifier, testFeatures);
predictedLabelsTree = predict(TreeClassifier, testFeatures);
predictedLabelsNB = predict(NaiveBayesClassifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels)

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2)) 

%disp('confMat 2 w/ fitclinear vs. fitcecoc')
%confMat2 = confusionmat(testLabels, predictedLabels2)
%confMat2 = bsxfun(@rdivide,confMat2,sum(confMat2,2))
disp('Tree Classifier')
confMat3 = confusionmat(testLabels, predictedLabelsTree)
Type_I_Error = confMat3(2) / (confMat3(2) + confMat3(1))
Type_II_Error = confMat3(3) / (confMat3(3) + confMat3(4))
Tree_Accuracy = (confMat3(1) + confMat3(4)) / (confMat3(1) + confMat3(2) + confMat3(3) + confMat3(4))
disp('Percentage Errors')
confMat3 = bsxfun(@rdivide, confMat3, sum(confMat3, 2))

disp('Naive Bayes Classifier')
confMat4 = confusionmat(testLabels,predictedLabelsNB)
Type_I_Error = confMat4(2) / (confMat4(2) + confMat4(1))
Type_II_Error = confMat4(3) / (confMat4(3) + confMat4(4))
NaiveBayes_Accuracy = (confMat4(1) + confMat4(4)) / (confMat4(1) + confMat4(2) + confMat4(3) + confMat4(4))
disp('Percentage Errors')
confMat4 = bsxfun(@rdivide, confMat4, sum(confMat4, 2))

%% 5.3: Test it on an unseen image
newImage = './doge.jpg'; % any cat or dog image should do!
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(classifier, imageFeatures);

% Display test image and assigned label
figure, imshow(img), title(char(label)); 

%% References
% [1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image
% database." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
% Conference on. IEEE, 2009.
%
% [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
% classification with deep convolutional neural networks." Advances in
% neural information processing systems. 2012.
%
% [3] Vedaldi, Andrea, and Karel Lenc. "MatConvNet-convolutional neural
% networks for MATLAB." arXiv preprint arXiv:1412.4564 (2014).
%
% [4] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding
% convolutional networks." Computer Vision-ECCV 2014. Springer
% International Publishing, 2014. 818-833.
%
% [5] Donahue, Jeff, et al. "Decaf: A deep convolutional activation feature
% for generic visual recognition." arXiv preprint arXiv:1310.1531 (2013).
