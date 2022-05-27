function txt = deepspeech2text(x,fs)
%deepspeech2text Transcribe speech to text using DeepSpeech
%    txt = deepspeech2text(audioIn,fs) processes audioIn through the
%    DeepSpeech model and returns a character vector corresponding to the
%    speech. Specify audioIn as a single-channel (mono) audio signal.
%    fs is the sampling rate of audioIn.
%
%    Example 1:
%        % Transcribe speech using speech-to-text
%        [audioIn,fs] = audioread("002.flac");
%        txt = deepspeech2text(audioIn,fs)
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
%       audioInTemplate = coder.typeof(zeros(10*16e3,1,'single'));
%       fsTemplate = coder.Constant(16e3);
%       argsTemplate = {audioInTemplate,fsTemplate};
%       codegen -config cfg -args argsTemplate deepspeech2text.m
%
%       % Test generated mex file
%       [audioIn,fs] = audioread("002.flac");
%       txt = deepspeech2text_mex(single(audioIn(1:16e3*10)),fs)
%
% See also deepspeech

%#codegen

persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork('deepspeech');
end

% Extract sequence of feature vectors from audio.
y = deepspeechFeatures(x,fs);
yb = deepspeechBuffer(y);

b = net.predict(yb);

% Convert predictions to a regular array.
c = gather(b);

% Decode predictions to a character vector.
txt = deepspeechPostprocess(c);

end