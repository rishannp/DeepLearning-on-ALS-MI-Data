%% Functional Connectivity
clear all
close all
clc

load Subject1.mat
S1 = Subject1;

doTraining = true;
threshold = 0.8;
adjacencymatrix.L = zeros([22,22,size(S1,2)]);
features.L = zeros([22,10,size(S1,2)]);

adjacencymatrix.R = zeros([22,22,size(S1,2)]);
features.R = zeros([22,10,size(S1,2)]);

adjacencymatrix.Re = zeros([22,22,size(S1,2)]);
features.Re = zeros([22,10,size(S1,2)]);

label = horzcat([ones(1,size(S1,2))],[2.*ones(1,size(S1,2))],[3.*ones(1,size(S1,2))]);

for i = 1:size(S1,2)
    plv = plvfcn(S1(i).L');
    graphObject = createPLVGraph(plv,threshold);
    scaledEdgeWeights = (graphObject.Edges.Weight - threshold) / ...
        (max(graphObject.Edges.Weight) - threshold);
    adjacencymatrix.L(:,:,i) = adjacency(graphObject);
    features.L(:,:,i) = bandp(S1(i).L);

    plv = plvfcn(S1(i).R');
    graphObject = createPLVGraph(plv,threshold);
    scaledEdgeWeights = (graphObject.Edges.Weight - threshold) / ...
        (max(graphObject.Edges.Weight) - threshold);
    adjacencymatrix.R(:,:,i) = adjacency(graphObject);
    features.R(:,:,i) = bandp(S1(i).R);

    plv = plvfcn(S1(i).Re');
    graphObject = createPLVGraph(plv,threshold);
    scaledEdgeWeights = (graphObject.Edges.Weight - threshold) / ...
        (max(graphObject.Edges.Weight) - threshold);
    adjacencymatrix.Re(:,:,i) = adjacency(graphObject);
    features.Re(:,:,i) = bandp(S1(i).Re);
   
end

adjacency = cat(3,adjacencymatrix.L,adjacencymatrix.R,adjacencymatrix.Re);
feature = cat(3,features.L,features.R,features.Re);
clear adjacencymatrix features plv S1 scaledEdgeWeights threshold graphObject

numGraphs = size(adjacency,3);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numGraphs,[0.3 0.3 0.4]);

featuresTrain = feature(:,:,idxTrain);
featuresValidation = feature(:,:,idxValidation);
featuresTest = feature(:,:,idxTest);

adjacencyTrain = adjacency(:,:,idxTrain);
adjacencyValidation = adjacency(:,:,idxValidation);
adjacencyTest = adjacency(:,:,idxTest);

labelsTrain = label(idxTrain);
labelsValidation = label(idxValidation);
labelsTest = label(idxTest);

% Data splitting
numFeatures = size(featuresTrain,2);
muX = zeros(1,numFeatures);
sigsqX = zeros(1,numFeatures);

for i = 1:numFeatures
    X = nonzeros(featuresTrain(:,i,:));
    muX(i) = mean(X);
    sigsqX(i) = var(X, 1);
end

numGraphsTrain = size(featuresTrain,3);

for j = 1:numGraphsTrain
    validIdx = 1:nnz(featuresTrain(:,1,j));
    featuresTrain(validIdx,:,j) = (featuresTrain(validIdx,:,j) - muX)./sqrt(sigsqX);
end

numGraphsValidation = size(featuresValidation,3);
for j = 1:numGraphsValidation
    validIdx = 1:nnz(featuresValidation(:,1,j));
    featuresValidation(validIdx,:,j) = (featuresValidation(validIdx,:,j) - muX)./sqrt(sigsqX);
end

classNames = [1,2,3];

TTrain = zeros(numGraphsTrain,numel(classNames));

for j = 1:numGraphsTrain
    if ~isempty(labelsTrain(j))
        [~,idx] = ismember(labelsTrain(j),classNames);
        TTrain(j,idx) = 1;
    end
end

classCounts = sum(TTrain,1);

figure
bar(classCounts)
ylabel("Count")
xticklabels(classNames)

labelCounts = sum(TTrain,2);

TValidation = zeros(numGraphsValidation,numel(classNames));
for j = 1:numGraphsValidation
    if ~isempty(labelsValidation(j))
        [~,idx] = ismember(labelsValidation(j),classNames);
        TValidation(j,idx) = 1;
    end
end

featuresTrain = arrayDatastore(featuresTrain,IterationDimension=3);
adjacencyTrain = arrayDatastore(adjacencyTrain,IterationDimension=3);
targetTrain = arrayDatastore(TTrain);

dsTrain = combine(featuresTrain,adjacencyTrain,targetTrain);

featuresValidation = arrayDatastore(featuresValidation,IterationDimension=3);
adjacencyValidation = arrayDatastore(adjacencyValidation,IterationDimension=3);
dsValidation = combine(featuresValidation,adjacencyValidation);

% Train
numHeads = struct;
numHeads.attn1 = 3;
numHeads.attn2 = 3;
numHeads.attn3 = 5;
parameters = struct;

numInputFeatures = size(feature,2);
numHiddenFeatureMaps = 96;
numClasses = numel(classNames);

sz = [numInputFeatures numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numInputFeatures;

parameters.attn1.weights.linearWeights = initializeGlorot(sz,numOut,numIn);
parameters.attn1.weights.attentionWeights = initializeGlorot([numOut 2],1,2*numOut);

sz = [numHiddenFeatureMaps numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numHiddenFeatureMaps;

parameters.attn2.weights.linearWeights = initializeGlorot(sz,numOut,numIn);
parameters.attn2.weights.attentionWeights = initializeGlorot([numOut 2],1,2*numOut);

numOutputFeatureMaps = numHeads.attn3*numClasses;

sz = [numHiddenFeatureMaps numOutputFeatureMaps];
numOut = numClasses;
numIn = numHiddenFeatureMaps;
parameters.attn3.weights.linearWeights = initializeGlorot(sz,numOut,numIn);
parameters.attn3.weights.attentionWeights = initializeGlorot([numOutputFeatureMaps 2],1,2*numOut);

parameters
parameters.attn1.weights

numEpochs = 100;
miniBatchSize = 30;
learnRate = 0.01;
labelThreshold = 0.5;
validationFrequency = 100;

mbq = minibatchqueue(dsTrain,4, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    OutputCast="double", ...
    OutputAsDlarray=[1 0 0 0], ...
    OutputEnvironment = ["auto" "cpu" "cpu" "cpu"]);

dsValidation.UnderlyingDatastores{1}.ReadSize = miniBatchSize;
dsValidation.UnderlyingDatastores{2}.ReadSize = miniBatchSize;

trailingAvg = [];
trailingAvgSq = [];

if doTraining

    % Initialize the training progress plot.
    figure
    C = colororder;
    
    lineLossTrain = animatedline(Color=C(2,:));
    lineLossValidation = animatedline( ...
        LineStyle="--", ...
        Marker="o", ...
        MarkerFaceColor="black");
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on

    iteration = 0;
    start = tic;
    
    % Loop over epochs.
    for epoch = 1:numEpochs

        % Shuffle data.
        shuffle(mbq);
            
        while hasdata(mbq)
            iteration = iteration + 1;
            
            % Read mini-batches of data.
            [XTrain,ATrain,numNodes,TTrain] = next(mbq);
    
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients,Y] = dlfeval(@modelLoss,parameters,XTrain,ATrain,numNodes,TTrain,numHeads);
            
            % Update the network parameters using the Adam optimizer.
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,learnRate);
            
            % Display the training progress.
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            loss = double(loss);
            addpoints(lineLossTrain,iteration,loss)
            drawnow
    
            % Display validation metrics.
            if iteration == 1 || mod(iteration,validationFrequency) == 0
                YValidation = modelPredictions(parameters,dsValidation,numHeads);
                lossValidation = crossentropy(YValidation,TValidation,ClassificationMode="multilabel",DataFormat="BC");

                lossValidation = double(lossValidation);
                addpoints(lineLossValidation,iteration,lossValidation)
                drawnow
            end
        end
    end
else
    % Load the pretrained parameters.
    display("Failed")
end

% Test

numGraphsTest = size(featuresTest,3);

for j = 1:numGraphsTest
    validIdx = 1:nnz(featuresTest(:,1,j));
    featuresTest(validIdx,:,j) = (featuresTest(validIdx,:,j) - muX)./sqrt(sigsqX);
end

featuresTest = arrayDatastore(featuresTest,IterationDimension=3,ReadSize=miniBatchSize);
adjacencyTest = arrayDatastore(adjacencyTest,IterationDimension=3,ReadSize=miniBatchSize);
dsTest = combine(featuresTest,adjacencyTest);

TTest = zeros(numGraphsTest,numel(classNames));

for j = 1:numGraphsTest
    if ~isempty(labelsTest(j))
        [~,idx] = ismember(labelsTest(j),classNames);
        TTest(j,idx) = 1;
    end
end

predictions = modelPredictions(parameters,dsTest,numHeads);
predictions = double(gather(extractdata(predictions)));
YTest = double(predictions >= 0.5);
scoreWeight = [0.5 1 2];
for i = 1:3
    scoreTest(i) = fScore(YTest,TTest,scoreWeight(i));
end
scoreTestTbl = table;
scoreTestTbl.Beta = scoreWeight';
scoreTestTbl.FScore = scoreTest'
figure
tiledlayout("flow")
for i = 1:numClasses
    nexttile
    confusionchart(YTest(:,i),TTest(:,i));
    title(classNames(i))
end

figure
tiledlayout("flow")

for i = 1:numClasses
    currentTargets = TTest(:,i)';
    currentPredictions = predictions(:,i)';

    [truePositiveRates,falsePositiveRates] = roc(currentTargets,currentPredictions);
    AUC = trapz(falsePositiveRates,truePositiveRates);

    nexttile
    plot(falsePositiveRates,truePositiveRates, ...
        falsePositiveRates,falsePositiveRates,"--",LineWidth=0.7)
    text(0.075,0.75,"\bf AUC = "+num2str(AUC),FontSize=6.75)
    xlabel("FPR")
    ylabel("TPR")
    title(classNames(i))
end

lgd = legend("ROC Curve - GAT", "ROC Curve - Random");
lgd.Layout.Tile = numClasses+1;

% YTest is mdl output, TTest is ground truth
% Calculate the number of samples
num_samples = size(YTest, 1);

% Initialize an array to store accuracies for each class
accuracies = zeros(1, size(YTest, 2));

% Iterate over each class
for class_idx = 1:size(YTest, 2)
    % Get the predicted outputs and ground truth labels for the current class
    predicted_class = YTest(:, class_idx);
    ground_truth_class = TTest(:, class_idx);
    
    % Count the number of correct predictions for the current class
    correct_predictions = sum(predicted_class == ground_truth_class);
    
    % Calculate the accuracy for the current class
    class_accuracy = correct_predictions / num_samples;
    
    % Store the accuracy for the current class
    accuracies(class_idx) = class_accuracy;
end

% Display the accuracies for each class
disp('Accuracies for each class:');
disp(accuracies);
%% Functions
function bp = bandp(x)

f = 0:4:40;
fs = 256;
bp = zeros(10,22);
for i = 1:size(f,2)-1
    freq = [f(i),f(i+1)];
    bp(i,:) = bandpower(x,fs,freq);
end
bp = bp';
end

function [Y,attentionScores] = model(parameters,X,A,numNodes,numHeads)

weights = parameters.attn1.weights;
numHeadsAttention1 = numHeads.attn1;

Z1 = X;
[Z2,attentionScores.attn1] = graphAttention(Z1,A,weights,numHeadsAttention1,"cat");
Z2  = elu(Z2);

weights = parameters.attn2.weights;
numHeadsAttention2 = numHeads.attn2;

[Z3,attentionScores.attn2] = graphAttention(Z2,A,weights,numHeadsAttention2,"cat");
Z3  = elu(Z3) + Z2;

weights = parameters.attn3.weights;
numHeadsAttention3 = numHeads.attn3;

[Z4,attentionScores.attn3] = graphAttention(Z3,A,weights,numHeadsAttention3,"mean");
Z4 = globalAveragePool(Z4,numNodes);

Y = sigmoid(Z4);

end

function [loss,gradients,Y] = modelLoss(parameters,X,adjacencyTrain,numNodes,T,numHeads)

Y = model(parameters,X,adjacencyTrain,numNodes,numHeads);
loss = crossentropy(Y,T,ClassificationMode="multilabel",DataFormat="BC");
gradients = dlgradient(loss,parameters);

end

function outFeatures = globalAveragePool(inFeatures,numNodes)

numGraphs = numel(numNodes);
numFeatures = size(inFeatures, 2);
outFeatures = zeros(numGraphs,numFeatures,"like",inFeatures);

startIdx = 1;
for i = 1:numGraphs
    endIdx = startIdx + numNodes(i) - 1;
    idx = startIdx:endIdx;
    outFeatures(i,:) = mean(inFeatures(idx,:));
    startIdx = endIdx + 1;
end

end

function y = elu(x)

y = max(0, x) + (exp(min(0, x)) -1);

end

function [outputFeatures,normAttentionCoeff] = graphAttention(inputFeatures,adjacency,weights,numHeads,aggregation)

% Split weights with respect to the number of heads and reshape the matrix to a 3-D array
szFeatureMaps = size(weights.linearWeights);
numOutputFeatureMapsPerHead = szFeatureMaps(2)/numHeads;
linearWeights = reshape(weights.linearWeights,[szFeatureMaps(1), numOutputFeatureMapsPerHead, numHeads]);
attentionWeights = reshape(weights.attentionWeights,[numOutputFeatureMapsPerHead, 2, numHeads]);

% Compute linear transformations of input features
value = pagemtimes(inputFeatures,linearWeights);

% Compute attention coefficients
query = pagemtimes(value, attentionWeights(:, 1, :));
key = pagemtimes(value, attentionWeights(:, 2, :));

attentionCoefficients = query + permute(key,[2, 1, 3]);
attentionCoefficients = leakyrelu(attentionCoefficients,0.2);

% Compute masked attention coefficients
mask = -10e9 * (1 - adjacency);
attentionCoefficients = attentionCoefficients + mask;

% Compute normalized masked attention coefficients
normAttentionCoeff = softmax(attentionCoefficients,DataFormat = "BCU");

% Normalize features using normalized masked attention coefficients
headOutputFeatures = pagemtimes(normAttentionCoeff,value);

% Aggregate features from multiple heads
if strcmp(aggregation, "cat")
    outputFeatures = headOutputFeatures(:,:);
else
    outputFeatures =  mean(headOutputFeatures,3);
end

end

function [features,adjacency,numNodes,target] = preprocessMiniBatch(featureData,adjacencyData,targetData)

% Extract feature and adjacency data from their cell array and concatenate the
% data along the third (batch) dimension
featureData = cat(3,featureData{:});
adjacencyData = cat(3,adjacencyData{:});

% Extract target data if it exists
if nargin > 2
    target = cat(1,targetData{:});
end

adjacency = sparse([]);
features = [];
numNodes = [];

for i = 1:size(adjacencyData, 3)
    % Get the number of nodes in the current graph
    numNodesInGraph = nnz(featureData(:,1,i));
    numNodes = [numNodes; numNodesInGraph];

    % Get the indices of the actual nonzero data
    validIdx = 1:numNodesInGraph;

    % Remove zero paddings from adjacencyData
    tmpAdjacency = adjacencyData(validIdx, validIdx, i);

    % Add self connections
    tmpAdjacency = tmpAdjacency + eye(size(tmpAdjacency));

    % Build the adjacency matrix into a block diagonal matrix
    adjacency = blkdiag(adjacency, tmpAdjacency);

    % Remove zero paddings from featureData
    tmpFeatures = featureData(validIdx, :, i);
    features = [features; tmpFeatures];
end

end

function score = fScore(predictions,targets,beta)

truePositive = sum(predictions .* targets,"all");
falsePositive = sum(predictions .* (1-targets),"all");

% Precision
precision = truePositive/(truePositive + falsePositive);

% Recall
recall = truePositive/sum(targets,"all");

% FScore
if nargin == 2
    beta = 1;
end

score = (1+beta^2)*precision*recall/(beta^2*precision+recall);

end

function Y = modelPredictions(parameters,ds,numHeads)

Y = [];

reset(ds)

while hasdata(ds)

    data = read(ds);

    featureData = data(:,1);
    adjacencyData = data(:,2);

    [features,adjacency,numNodes] = preprocessMiniBatch(featureData,adjacencyData);

    X = dlarray(features);

    minibatchPred = model(parameters,X,adjacency,numNodes,numHeads);
    Y = [Y;minibatchPred];
end

end

function plottopo(x)
image_path = '10202.png';  % Replace with the path to your image
img = imread(image_path);
imshow(img);
hold on;

elecpos = {
    'FP1', [205, 87];
    'FP2', [294, 90];
    'F7',  [131, 143];
    'F3',  [188, 155];
    'Fz',  [252, 159];
    'F4',  [313, 155];
    'F8',  [373, 141];
    'T7',  [104, 236];
    'C3',  [176, 235];
    'Cz',  [249, 232];
    'C4',  [325, 234];
    'T8',  [401, 235];
    'P7',  [130, 327];
    'P3',  [187, 314];
    'Pz',  [254, 310];
    'P4',  [314, 312];
    'P8',  [372, 326];
    'O1',  [205, 382];
    'O2',  [295, 379];
    'lEOG',  [167,67];
    'cEOG',  [250,48];
    'rEOG',  [315,60];
};

loc = cell2mat(elecpos(:,2));

% Display the graph with specified node positions and edge weights
colormap hot;
plot(x.graphObject, 'XData', loc(:, 1), 'YData', loc(:, 2), ...
     'EdgeCData', x.scaledEdgeWeights, 'LineWidth', 2.5);
colorbar;
hold off;
end

function [adjacency_matrix, local_clustering] = diradjclust(graphObject)
    % Get the number of nodes
    num_nodes = numnodes(graphObject);

    % Initialize adjacency matrix and local clustering coefficient vector
    adjacency_matrix = zeros(num_nodes);
    local_clustering = zeros(num_nodes, 1);

    % Create the adjacency matrix
    adjacency_matrix = adjacency(graphObject, 'weighted');

    % Calculate the local clustering coefficient for each node
    for i = 1:num_nodes
        neighbor = neighbors(graphObject, i, 'outgoing');  % Get outgoing neighbors of node i
        num_neighbors = length(neighbor);
        
        if num_neighbors <= 1
            local_clustering(i) = 0;  % Avoid division by zero for isolated nodes
        else
            connected_edges = sum(adjacency_matrix(neighbor, neighbor), 'all') / 2;
            possible_edges = num_neighbors * (num_neighbors - 1) / 2;
            local_clustering(i) = connected_edges / possible_edges;
        end
    end
end

function [adjacency_matrix, local_clustering] = adjclust(graphObject)
    % Get the number of nodes
    num_nodes = numnodes(graphObject);

    % Initialize adjacency matrix and local clustering coefficient vector
    adjacency_matrix = zeros(num_nodes);
    local_clustering = zeros(num_nodes, 1);

    % Create the adjacency matrix
    adjacency_matrix = adjacency(graphObject);

    % Calculate the local clustering coefficient for each node
    for i = 1:num_nodes
        neighbor = neighbors(graphObject, i);  % Get neighbors of node i
        num_neighbors = length(neighbor);
        
        if num_neighbors <= 1
            local_clustering(i) = 0;  % Avoid division by zero for isolated nodes
        else
            connected_edges = sum(adjacency_matrix(neighbor, neighbor), 'all') / 2;
            possible_edges = num_neighbors * (num_neighbors - 1) / 2;
            local_clustering(i) = connected_edges / possible_edges;
        end
    end
end

function [G,Gd] = createPLVGraph(phaseLockingValues, threshold)
    numElectrodes = size(phaseLockingValues, 1);

    % Create a graph object
    G = graph();
    Gd = digraph();

    % Add nodes to the graph
    G = addnode(G, numElectrodes);
    Gd = addnode(G, numElectrodes);
    
    % Iterate over the phase locking values matrix and add weighted edges
    for i = 1:numElectrodes
        for j = i+1:numElectrodes
            plv = phaseLockingValues(i, j);
            if plv > threshold  % Add edges only if PLV is above the threshold
                G = addedge(G, i, j, plv);
                Gd = addedge(G, i, j, plv);
            end
        end
    end
end

function plvMatrix = plvfcn(eegData)

numElectrodes = size(eegData, 1);
numTimeSteps = size(eegData, 2);

% Initialize the PLV matrix
plvMatrix = zeros(numElectrodes, numElectrodes);

% Calculate PLV matrix for all electrode pairs
for electrode1 = 1:numElectrodes
    for electrode2 = electrode1+1:numElectrodes
        % Calculate the instantaneous phase of the signals
        phase1 = angle(hilbert(eegData(electrode1, :)));
        phase2 = angle(hilbert(eegData(electrode2, :)));
        
        % Calculate the phase difference between the two signals
        phase_difference = phase2 - phase1;
        
        % Calculate the Phase-Locking Value (PLV)
        plv = abs(sum(exp(1i * phase_difference)) / numTimeSteps);
        
        % Store the PLV value in the matrix (both upper and lower triangular)
        plvMatrix(electrode1, electrode2) = plv;
        plvMatrix(electrode2, electrode1) = plv;
    end
end

end

function [trainacc, testacc, mean1] = AAFC(t,winlen,S9,window)


x = 0:t:size(S9,1);
x(end) = length(S9); % Data splits (5)
accuracy = [0;0;0];

for i = 1:length(x)-2
    
    if i == 1
        ltr = S9([x(i)+1:x(i+1)],:);
        rtr = S9([x(i)+1:x(i+1)],:);
        retr = S9([x(i)+1:x(i+1)],:);
        
        for j = 1:size(ltr,1)
            for k = 1:size(ltr,2)
                S9(j,k).fcL = plvfcn(S9(j,k).L);
                S9(j,k).fcR = plvfcn(S9(j,k).R);
                S9(j,k).fcRe = plvfcn(S9(j,k).Re);
            end
        end
        
        
        % Accuracy var is L,R,Re in order of rows.
        id = size(accuracy,2); % update accuracy end
        [mdl.L, accuracy(1,id+1)] = trainQDA(lTrain, lLabel); % 5 Fold CV
        [mdl.R, accuracy(2,id+1)] = trainQDA(rTrain, rLabel);
        [mdl.Re, accuracy(3,id+1)] = trainQDA(reTrain, reLabel);
        
        test = S9([x(i+1)+1:x(i+2)],:); % next piece of testing data
        
        [l.test,l.label] = Ltest1(test,Wl,2,lBand); % finding feature vector of l/r/re
        [re.test,re.label] = Retest1(test,Wre,2,reBand);
        [r.test,r.label] = Rtest1(test,Wr,2,rBand);
        
        out.l = mdl.L.predictFcn(l.test);  % make predictions based on previous trained model
        out.r = mdl.R.predictFcn(r.test);
        out.re = mdl.Re.predictFcn(re.test);
        
        id = size(accuracy,2); % update accuracy end
        l.label == out.l; %% Test
        accuracy(1,id+1) = (sum(ans)/length(ans)) ;
        correct.l = ans;
        
        r.label == out.r;
        accuracy(2,id+1) = (sum(ans)/length(ans)) ;
        correct.r = ans;
        
        re.label == out.re;
        accuracy(3,id+1) = (sum(ans)/length(ans)) ;
        correct.re = ans;
        
        idxl = [];
        idxr = [];
        idxre = [];
        for j = 1:length(correct.l)/2
            if correct.l(j) == 1
                %lTrain = vertcat(lTrain,l.test(j,:));
                %lLabel = vertcat(lLabel,l.label(j));
                idxl = horzcat(idxl,x(i)+j);
                if idxl < 5
                    ltr = vertcat(ltr,test(j,:));
                end
                if j > 5
                    ltr = vertcat(ltr,test([j-5],:));
                end
            end
            
            if correct.r(j) == 1
                %rTrain = vertcat(rTrain,r.test(j,:));
                %rLabel = vertcat(rLabel,r.label(j));
                idxr = horzcat(idxr,x(i)+j);
                if j < 5
                    rtr = vertcat(rtr,test(j,:));
                end
                if j > 5
                    rtr = vertcat(rtr,test([j-5],:));
                end
                
            end
            
            if correct.re(j) == 1
                %reTrain = vertcat(reTrain,re.test(j,:));
                %reLabel = vertcat(reLabel,re.label(j));
                idxre = horzcat(idxre,x(i)+j);
                if j < 5
                    retr = vertcat(retr,test(j,:));
                end
                if j > 5
                    retr = vertcat(retr,test([j-5],:));
                end
            end
        end
        
        
        [reBand,Wre,reTrain,reLabel] = fbcspRe(retr,2); % Retrain the spatial filter with the additional correct trials from testing
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr,2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(rtr,2);
        
        id = size(accuracy,2); % update accuracy end
        [mdl.L, accuracy(1,id+1)] = trainQDA(lTrain, lLabel); % Retrain classifier
        [mdl.R, accuracy(2,id+1)] = trainQDA(rTrain, rLabel);
        [mdl.Re, accuracy(3,id+1)] = trainQDA(reTrain, reLabel);
        
        %         figure()
        %         subplot(3,1,1)
        %         gscatter(lTrain(:,1),lTrain(:,2),lLabel);
        %         subplot(3,1,2)
        %         gscatter(rTrain(:,1),rTrain(:,2),rLabel);
        %         subplot(3,1,3)
        %         gscatter(reTrain(:,1),reTrain(:,2),reLabel);
        
    end
    
    test = S9([x(i+1)+1:x(i+2)],:); % next piece of testing data
    
    [l.test,l.label] = Ltest1(test,Wl,2,lBand); % finding feature vector of l/r/re
    [re.test,re.label] = Retest1(test,Wre,2,reBand);
    [r.test,r.label] = Rtest1(test,Wr,2,rBand);
    
    %     figure()
    %     subplot(3,1,1)
    %     gscatter(l.test(:,1),l.test(:,2),l.label);
    %     subplot(3,1,2)
    %     gscatter(r.test(:,1),r.test(:,2),r.label);
    %     subplot(3,1,3)
    %     gscatter(re.test(:,1),re.test(:,2),re.label);
    
    out.l = mdl.L.predictFcn(l.test);  % make predictions based on previous trained model
    out.r = mdl.R.predictFcn(r.test);
    out.re = mdl.Re.predictFcn(re.test);
    
    id = size(accuracy,2); % update accuracy end
    l.label == out.l; %% Test
    accuracy(1,id+1) = (sum(ans)/length(ans)) ;
    correct.l = ans;
    
    r.label == out.r;
    accuracy(2,id+1) = (sum(ans)/length(ans)) ;
    correct.r = ans;
    
    re.label == out.re;
    accuracy(3,id+1) = (sum(ans)/length(ans)) ;
    correct.re = ans;
    
    idxl = [];
    idxr = [];
    idxre = [];
    for j = 1:length(correct.l)/2
        if correct.l(j) == 1
            %lTrain = vertcat(lTrain,l.test(j,:));
            %lLabel = vertcat(lLabel,l.label(j));
            idxl = horzcat(idxl,x(i)+j);
            if idxl < 5
                ltr = vertcat(ltr,test(j,:));
            end
            if j > 5
                ltr = vertcat(ltr,test([j-5],:));
            end
        end
        
        if correct.r(j) == 1
            %rTrain = vertcat(rTrain,r.test(j,:));
            %rLabel = vertcat(rLabel,r.label(j));
            idxr = horzcat(idxr,x(i)+j);
            if j < 5
                rtr = vertcat(rtr,test(j,:));
            end
            if j > 5
                rtr = vertcat(rtr,test([j-5],:));
            end
            
        end
        
        if correct.re(j) == 1
            %reTrain = vertcat(reTrain,re.test(j,:));
            %reLabel = vertcat(reLabel,re.label(j));
            idxre = horzcat(idxre,x(i)+j);
            if j < 5
                retr = vertcat(retr,test(j,:));
            end
            if j > 5
                retr = vertcat(retr,test([j-5],:));
            end
        end
    end
    
    if window == 1
        [reBand,Wre,reTrain,reLabel] = fbcspRe(retr([end-winlen:end],:),2); % Retrain the spatial filter with the additional correct trials from testing
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr([end-winlen:end],:),2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(rtr([end-winlen:end],:),2);
    elseif window == 0
        [reBand,Wre,reTrain,reLabel] = fbcspRe(retr,2); % Retrain the spatial filter with the additional correct trials from testing
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr,2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(rtr,2);
    end
    
    %     figure()
    %     subplot(3,1,1)
    %     gscatter(lTrain(:,1),lTrain(:,2),lLabel);
    %     subplot(3,1,2)
    %     gscatter(rTrain(:,1),rTrain(:,2),rLabel);
    %     subplot(3,1,3)
    %     gscatter(reTrain(:,1),reTrain(:,2),reLabel);
    
    id = size(accuracy,2); % update accuracy end
    [mdl.L, accuracy(1,id+1)] = trainQDA(lTrain, lLabel); % Retrain classifier
    [mdl.R, accuracy(2,id+1)] = trainQDA(rTrain, rLabel);
    [mdl.Re, accuracy(3,id+1)] = trainQDA(reTrain, reLabel);
end

%
% Find columns with all zeros
zero_columns = all(accuracy == 0);

% Delete columns from the matrix
accuracy(:, zero_columns) = [];

testacc = accuracy(:,[2:2:length(accuracy)]);
trainacc = accuracy(:,[1:2:length(accuracy)]);

[mean1] = fig(trainacc, testacc);

end