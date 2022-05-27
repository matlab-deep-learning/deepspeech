function output = deepspeechPostprocess(Y,varargin)
%deepspeechPostprocess Decode output from DeepSpeech network
%    txt = deepspeechPostprocess(probsSeq) decodes the sequence of
%    probability vectors into text. probsSeq is the output from a
%    DeepSpeech network.
%
%    txt = deepspeechPostprocess(probsSeq,dict) uses a non-default
%    dictionariy, dict, to decode the probability sequence into text.
%    Specify dict as a cell array with the same number of elements as
%    classes in probsSeq.
%
%    txt =  deepspeechPostprocess(probsSeq,dict,blankIndex) specifies which
%    element of dict corresponds to the blank token in CTC training. If
%    unspecified, the last element of dict is assumed to be the blank
%    token.
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
% See also speech2text, deepspeech, deepspeechPreprocess

%#codegen

if nargin < 2
    dict = {' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p', ...
        'q','r','s','t','u','v','w','x','y','z','''','<blank>'};
else
    dict = varargin{1};
end

if nargin<3
    blankIdx = numel(dict);
else
    blankIdx = varargin{2};
end

[~,argmax] = max(Y,[],1);
argmax = argmax(:);

txt = cell(numel(argmax),1);
for ii = 1:numel(txt)
    txt{ii} = dict{argmax(ii)};
end

% Remove repeats
isChangePoint = diff([0;argmax])~=0;
changePointIdx = find(isChangePoint);
numChangePoint = numel(changePointIdx);
txt2 = cell(1,numChangePoint);
for ii = 1:numChangePoint
    txt2{ii} = txt{changePointIdx(ii)};
end

% Join string
txt3 = strjoin(txt2,'');

% Remove blank tokens
output = erase(txt3,dict{blankIdx});

end