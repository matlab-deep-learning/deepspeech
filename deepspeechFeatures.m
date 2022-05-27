function coeffs = deepspeechFeatures(audioIn,fs)
%deepspeechFeatures Extract features for DeepSpeech network
%    features = deepspeechFeatures(audioIn,fs) generates mel-frequency
%    cepstral coefficients from the audio input, audioIn, that can be fed
%    to the DeepSpeech pretrained network. fs is the sampling rate, in Hz.
%    features is returned as an M-by-T array, where M is the number of
%    coefficients (26), and T is the number of analysis windows over time.
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
% See also deepspeechBuffer, deepspeech, deepspeechPostprocess

%#codegen

% Copyright 2022 The MathWorks, Inc.

% Define DeepSpeech parameters
fs0 = 16e3;
windowLength = 512;
fftLength = windowLength;
halfsidedSpectrum = 257;
overlapLength = windowLength - 320;
numCoeffs = 26;
numBands = 40;
frequencyRange = [20,8000];

persistent deepspeechfilterbank
if isempty(deepspeechfilterbank)
    deepspeechfilterbank = iDesignDeepSpeechFilterBank(fs0,frequencyRange,numBands,fftLength);
end

% Validate inputs
validateattributes(audioIn,{'single','double'},{'column','nonnan','finite'},'deepspeechFeatures','audioIn')
validateattributes(fs,{'single','double'},{'scalar','nonnan','finite'},'deepspeechFeatures','fs')

% Resample the audio if appropriate and cast to single
if fs0~=fs
    x = cast(resample(double(audioIn),fs0,double(fs)),'single');
else
    x = single(audioIn);
end

% Perform short-time Fourier transform
xb = deepspeechUtility.buffer(x,windowLength,windowLength-overlapLength);
xbw = bsxfun(@times,xb,hann(windowLength,"periodic"));
Y = fft(xbw,fftLength);
Yhalf = Y(1:halfsidedSpectrum,:);
S = sqrt(real(Yhalf.*conj(Yhalf)));

% Compute auditory spectrogram                               
melS = deepspeechfilterbank*S;

% Extract cepstral coefficients
melS = log(melS);
[L,M,N] = size(melS);
melS = reshape(melS,L,M*N);
DCTmatrix = deepspeechUtility.createDCTMatrix(numCoeffs,L,underlyingType(audioIn));% Design DCT matrix
coeffs = DCTmatrix * melS; % Apply DCT matrix
coeffs = reshape(coeffs,numCoeffs,M,N); % Unpack matrix back to 3d array
coeffs = permute(coeffs,[2 1 3]); % Put time along the first dimension

% Scale DCT to match original implementation
coeffs(:,1) = sqrt(2) * coeffs(:,1);

coeffs = permute(coeffs,[2,1,3]);

end

%% Design DeepSpeech Filter Bank
function fb = iDesignDeepSpeechFilterBank(fs,frequencyRange,numBands,fftLength)
%iDesignDeepSpeechFilterBank Design DeepSpeech-style mel filterbank
% fb = iDesignDeepSpeechFilterBank(fs,frequencyRange,numBands,fftLength)
% designs a DeepSpeech-style mel filter bank according to the parameters.

% Determine the number of bins in a half-sided spectrum
if mod(fftLength,2) == 0
    numBins = fftLength/2+1;
else
    numBins = (fftLength+1)/2;
end

% Convert frequency range to mel
melLow = hz2mel(frequencyRange(1));
melHigh = hz2mel(frequencyRange(2));

% Get center frequency of bandpass filters, in mel.
melSpan = melHigh - melLow;
melSpacing = melSpan / (numBands + 1);
cf = zeros(1,numBands+1);
for ii = 0:numBands
    cf(ii+1) = melLow + (melSpacing * (ii + 1));
end

hzPerBin = 0.5*fs/(numBins - 1);
startIndex = floor(1.5 + frequencyRange(1)/hzPerBin);
endIndex = floor(frequencyRange(2)/hzPerBin);

% Maps the input spectrum bin indices to filter bank bands. For each FFT
% bin, bandMap indicates which band this bin contributes to on the right
% side of the triangle.  Thus this bin also contributes to the left side of
% the next band's triangle response.
aBand = 0;
bandMap = zeros(1,numBins);
for ii = 0:numBins-1
    melf = hz2mel(ii*hzPerBin );
    if (ii < startIndex) || (ii > endIndex)
        bandMap(ii+1) = -2;  % Indicate an unused Fourier coefficient.
    else
        while (aBand < numBands) && (cf(aBand+1) < melf)
            aBand = aBand + 1;
        end
        bandMap(ii+1) = aBand - 1;  % Can be == -1
    end
end

%  The contribution of any one FFT bin is based on its distance between two
%  mel-channel center frequencies.  This bin contributes weights(ii) to the
%  current channel and 1-weights(ii) to the next channel. Create a
%  weighting function to taper the band edges.
weights = zeros(1, numBins);
for ii = 0:numBins-1
    aBand = bandMap(ii+1);
    if (ii < startIndex) || (ii > endIndex)
        weights(ii+1) = 0.0;
    else
        if (aBand >= 0)
            weights(ii+1) =...
                (cf(aBand + 2) - hz2mel(ii * hzPerBin)) / ...
                (cf(aBand + 2) - cf(aBand+1));
        else
            weights(ii+1) = (cf(1) - hz2mel(ii * hzPerBin)) /...
                (cf(1) - melLow);
        end
    end
end

% Create filter bank
fb = zeros(numBands,numBins);
for ii = startIndex:endIndex
    aBand = bandMap(ii+1);

    % Right side of triangle, downward slope
    if aBand >= 0
        fb(aBand+1,ii+1) = weights(ii+1);
    end
    
    % Left side of triangle
    aBand = aBand + 1;
    if aBand < numBands
        fb(aBand+1,ii+1) = 1 - weights(ii+1);
    end
end

end
