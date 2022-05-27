function features = deepspeechBuffer(x)
%deepspeechBuffer Buffer features for DeepSpeech network
%   features = deepspeechBuffer(features) buffers the features output from
%   deepspeechFeatures to a C-by-T-by-B array, where C is the buffered
%   feature vector (494), T is the time dimension, and B is the batch
%   dimension.
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
%        net = deepspeech();
%
%        % Pass features through network.
%        y = predict(net,features);
%
%        % Decode predictions.
%        txt = deepspeechPostprocess(y)
%
% See also deepspeech, deepspeechPostprocess, deepspeechFeatures

%#codegen

% Copyright 2022 The MathWorks, Inc.

% Define DeepSpeech parameters
windowWidth = 19;

% Buffer mfcc coeffs to match original implementation
numCoeffs = size(x,1);
numSignals = size(x,3);

if rem(windowWidth,2)~=0
    padFront = (numCoeffs/2)*(windowWidth - 1);
    padBack = padFront;
else
    padFront = (numCoeffs/2)*(windowWidth - 2);
    padBack = (numCoeffs/2)*(windowWidth);
end

x = reshape(x,size(x,1)*size(x,2),size(x,3));
x = cat(1,zeros(padFront,numSignals,like=x),x,zeros(padBack,numSignals,like=x));
featuresBuffered = deepspeechUtility.buffer(x,windowWidth*numCoeffs,windowWidth*numCoeffs-(numCoeffs*(windowWidth-1)));
features = reshape(featuresBuffered,size(featuresBuffered,1),[],numSignals);

end