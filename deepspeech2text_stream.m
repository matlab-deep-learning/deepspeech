function txt = deepspeech2text_stream(x,fs,varargin)
%deepspeech2text_stream Transcribe speech to text using DeepSpeech
%    txt = deepspeech2text_stream(audioIn,fs) processes audioIn through the
%    DeepSpeech model and returns a character vector corresponding to the
%    speech. Specify audioIn as a single-channel (mono) audio signal.
%    fs is the sampling rate of audioIn.
%
%    txt = deepspeech2text(audioIn,fs,Reset=TF) returns the network to
%    initial values before calling predictAndUpdateState. By default, Rest
%    is set to false.
%
%    Example 1:
%       % Transcribe speech using speech-to-text
%       [audioIn,fs] = audioread("002.flac");
%       txt1 = deepspeech2text_stream(single(audioIn(1:5*fs)),fs,Reset=true);
%       txt2 = deepspeech2text_stream(single(audioIn(5*fs+1:10*fs)),fs);
%       txt = [txt1,txt2]
%
%    Example 2:
%       % Generate C code to perform speech-to-text transcription.
%
%       % Define code generation configuration
%       cfg = coder.config('mex');
%       cfg.TargetLang = 'C';
%       cfg.DeepLearningConfig = coder.DeepLearningConfig('none');
%
%       % Generate code
%       audioInTemplate = coder.typeof(zeros(5*16e3,1,'single'));
%       fsTemplate = coder.Constant(16e3);
%       resetTemplate = true;
%       argsTemplate = {audioInTemplate,fsTemplate,'Reset',resetTemplate};
%       codegen -config cfg -args argsTemplate deepspeech2text_stream.m
%
%       % Test generated mex file
%       [audioIn,fs] = audioread("002.flac");
%       txt1 = deepspeech2text_stream_mex(single(audioIn(1:5*fs)),fs,'Reset',true);
%       txt2 = deepspeech2text_stream_mex(single(audioIn(5*fs+1:10*fs)),fs,'Reset',false);
%       txt = [txt1,txt2]
%
% See also deepspeech

%#codegen

persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork('deepspeech');
end

if nargin==4
    if strcmpi(varargin{1},"Reset") && varargin{2}
        net = resetState(net);
    end
end

% Extract sequence of feature vectors from audio.
y = deepspeechFeatures(x,fs);
yb = deepspeechBuffer(y);

[net,b] = net.predictAndUpdateState(yb);

% Convert predictions to a regular array.
c = gather(b);

% Decode predictions to a character vector.
txt = deepspeechPostprocess(c);

end