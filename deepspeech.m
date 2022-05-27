function [net,dict] = deepspeech(varargin)
%deepspeech Load DeepSpeech speech2text network
%    net = deepspeech() returns a pretrained DeepSpeech
%    network as a DAGNetwork.
%
%    net = deepspeech(PreTrained=TF) specifies whether to return a trained
%    or untrained DeepSpeech network. If untrained, the network is returned
%    as a layer graph. PreTrained defaults to true.
%
%    [...,dict] = deepspeech() also returns the dictionary required to
%    decode predictions.
%
%    Example:
%        % Use DeepSpeech to perform speech-to-text transcription.
%
%        % Read audio signal.
%        [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");
%
%        % Extract features from audio signal.
%        features = deepspeechFeatures(audioIn,fs);
%
%        % Buffer features.
%        features = deepspeechBuffer(features);
%
%        % Create DeepSpeech network.
%        net = deepspeech;
%
%        % Pass features through network.
%        y = predict(net,features);
%
%        % Decode predictions.
%        txt = deepspeechPostprocess(y)
%
% See also deepspeech2text, deepspeechPostprocess, deepSpeechFeatures

% Copyright 2022 The MathWorks, Inc.

% Input checks
narginchk(0,2)
if nargin>0
    validatestring(varargin{1},"PreTrained","deepspeech","PreTrained");
end
if nargin==1
    error("Invalid number of arguments.")
end
if nargin==2
    validateattributes(varargin{2},{'numeric','logical'},"scalar","deepspeech","PreTrained")
    isPretrained = logical(varargin{2});
else
    isPretrained = true;
end

if isPretrained && ~exist('deepspeechWeights.mat','file')
    error("deepspeech:ModelNotFound", ...
        "Unable to access the pretrained weights.\nMake sure the required files are installed.\n" + ...
        "Download https://ssd.mathworks.com/supportfiles/audio/deepspeech/deepspeech.zip and then unzip the file to a location on the MATLAB path.")
else
    % Load the weights and biases
    load("deepspeechWeights.mat","learnables");
end

% DeepSpeech parameters
windowWidth = 19;
numCoeffs = 26;
inputSize = numCoeffs*windowWidth;
numClasses = 29;
numHiddenUnits = 2048;
clipVal = 20;
dropoutRate = 0.05;
dict = {' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p', ...
    'q','r','s','t','u','v','w','x','y','z','''','<blank>'};

% Create the layer graph
if isPretrained
    layers = [ ...
        sequenceInputLayer(inputSize,Name="input")

        fullyConnectedLayer(numHiddenUnits,Name="fc1",Weights=learnables.fc1.Weights,Bias=learnables.fc1.Bias)
        clippedReluLayer(clipVal,Name="clip1")
        dropoutLayer(dropoutRate,Name="dropout1")

        fullyConnectedLayer(numHiddenUnits,Name="fc2",Weights=learnables.fc2.Weights,Bias=learnables.fc2.Bias)
        clippedReluLayer(clipVal,Name="clip2")
        dropoutLayer(dropoutRate,Name="dropout2")

        fullyConnectedLayer(numHiddenUnits,Name="fc3",Weights=learnables.fc3.Weights,Bias=learnables.fc3.Bias)
        clippedReluLayer(clipVal,Name="clip3")
        dropoutLayer(dropoutRate,Name="dropout3")

        lstmLayer(numHiddenUnits,Name="lstm",InputWeights=learnables.lstm.InputWeights,RecurrentWeights=learnables.lstm.RecurrentWeights,Bias=learnables.lstm.Bias)

        fullyConnectedLayer(numHiddenUnits,Name="fc4",Weights=learnables.fc4.Weights,Bias=learnables.fc4.Bias)
        clippedReluLayer(clipVal,Name="clip4")
        dropoutLayer(dropoutRate,Name="dropout4")

        fullyConnectedLayer(numClasses,Name="fc5",Weights=learnables.fc5.Weights,Bias=learnables.fc5.Bias)
        softmaxLayer(Name="softmax")];

    lg = layerGraph(layers);

    lg = addLayers(lg,classificationLayer(Name="classOutput",Classes=categorical(1:numel(dict))));
    lg = connectLayers(lg,'softmax','classOutput');
    net = assembleNetwork(lg);
else
    layers = [ ...
        sequenceInputLayer(inputSize,Name="input")

        fullyConnectedLayer(numHiddenUnits,Name="fc1")
        clippedReluLayer(clipVal,Name="clip1")
        dropoutLayer(dropoutRate,Name="dropout1")

        fullyConnectedLayer(numHiddenUnits,Name="fc2")
        clippedReluLayer(clipVal,Name="clip2")
        dropoutLayer(dropoutRate,Name="dropout2")

        fullyConnectedLayer(numHiddenUnits,Name="fc3")
        clippedReluLayer(clipVal,Name="clip3")
        dropoutLayer(dropoutRate,Name="dropout3")

        lstmLayer(numHiddenUnits,Name="lstm")

        fullyConnectedLayer(numHiddenUnits,Name="fc4")
        clippedReluLayer(clipVal,Name="clip4")
        dropoutLayer(dropoutRate,Name="dropout4")

        fullyConnectedLayer(numClasses,Name="fc5")
        softmaxLayer(Name="softmax")];
    net = layerGraph(layers);
end

end

