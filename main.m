% you need to add folder music_speech and its subfolders in path

% createing path for music and speech files
files_music = dir('music_speech/music_wav/*.wav');
files_speech = dir('music_speech/speech_wav/*.wav');

% counter for indexing matrix
count1 = 1;
for fileM = files_music'
    % read .wav and save to temp file
    [musicTemp, fs] = audioread(fileM.name);
    %music_data(:,count1) = musicTemp(:,1);
    
    %use temp data for melcepst with num of frames and write as matrix
    music_melcepst{count1} = melcepst(musicTemp, fs);
    count1 = count1 + 1;
end

count2 = 1;     
for fileS = files_speech'
    speechTemp = audioread(fileS.name);
    %speech_data(:,count2) = speechTemp(:,1);
    speech_melcepst{count2} = melcepst(speechTemp(:,1), fs);
    count2 = count2 + 1;
end

melRows = count1 - 1;
melMSize = max(size(music_melcepst{1}));
melSSize = max(size(speech_melcepst{1}));

startPos = 1;
for n = 1:melRows
    for m = 1:melMSize
       music_speech_meas(startPos,:) = music_melcepst{n}(m,:);
       startPos = startPos+1;
    end 
end
for n = 1:melRows
    for m = 1:melSSize
       music_speech_meas(startPos,:) = speech_melcepst{n}(m,:);
       startPos = startPos+1;
    end 
end

% create type matrix
sizeOfMeas = max(size(music_speech_meas));
soundTypes = cell(sizeOfMeas,1);
halfMeasFor = sizeOfMeas/2+1;
for i=1:halfMeasFor
    soundTypes{i} = 'music';
end
for i=halfMeasFor:sizeOfMeas
    soundTypes{i} = 'speech';
end

statOfTypes = tabulate(soundTypes);

lmHalf = i/2;
lmeasM = i/2*0.75;       
lmeasS = lmHalf + lmeasM;

ms_meas = music_speech_meas([1:lmeasM lmHalf:lmeasS],:);
ms_type = soundTypes([1:lmeasM lmHalf:lmeasS],:);

test_meas = music_speech_meas([lmeasM:lmHalf lmeasS:i],:);
test_type = soundTypes([lmeasM:lmHalf lmeasS:i],:);

%learn bayes classificator
O1 = fitNaiveBayes(ms_meas,ms_type);

%test bayes classificator
C1 = O1.predict(test_meas);
cMat1 = confusionmat(test_type,C1);

msSvmMeas = music_speech_meas([1:20000 170000:190000],:);
msSvmType = soundTypes([1:20000 170000:190000],:);

%learn svm classificatior
SVMModel = fitcsvm(msSvmMeas, msSvmType, 'KernelFunction', 'linear');

%test svm classificator
[svmLabel,svmScore] = predict(SVMModel, test_meas);
cMat2 = confusionmat(test_type, svmLabel);


figure
gscatter(music_speech_meas(:,1),music_speech_meas(:,12),soundTypes);
hold off
