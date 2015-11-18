function offlineanalysis(vidno,modelno)


patchSize=50;
testFrames=[];

userDataStruct.currentImageIdx=2;
userDataStruct.TrainData=[];
userDataStruct.modelstruct=[];

% Read input
%[filename filepath] = uigetfile('*.avi');
%filename=[filepath filename];
if vidno==2
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_1/2013_05_30_Photodamage_15-55_kW_ungefaerbt_P3_10x_Ph.avi';
elseif vidno==7
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_1/2013_05_30_Photodamage_15-55kW_ungefaerbt_P4_10x_Ph.avi';
elseif vidno==8
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_1/2013_05_30_Photodamage_15-55kW_ungefaerbt_P7_10x_Ph.avi';
elseif vidno==9
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_1/2013_05_30_Photodamage_15-55kW_ungefaerbt_P9_10x_Ph.avi';    
elseif vidno==10
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_2/2013_06_04_Photodamage_50-100kW_ungefaerbt_P5_10x_Ph.avi'; 
elseif vidno==11
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_2/2013_06_04_Photodamage_50-100kW_ungefaerbt_P7_10x_Ph.avi';
elseif vidno==12
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_2/2013_06_04_Photodamage_50-100kW_ungefaerbt_P8_10x_Ph.avi';
elseif vidno==13
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_2/2013_06_04_Photodamage_50-100kW_ungefaerbt_P15_10x_Ph.avi';    
elseif vidno==14
    filename='/home/mkandemi/Documents/data/zeiss/Filme_geschaedigte_Zellen_2/2013_06_04_Photodamage_50-100kW_ungefaerbt_P16_10x_Ph.avi';        
end
fprintf('%s\n',filename);
  
vid=VideoReader(filename);
N=vid.NumberOfFrames;
userDataStruct.Video=read(vid,[1 N]);

% Extract features
for ii=2:5
    fprintf('Frame: %d/%d is being processed\n',ii,N);
    if modelno <= 4
      features=extract_frame_features(userDataStruct.Video(:,:,:,ii),userDataStruct.Video(:,:,:,ii-1),ii,patchSize);
    else % if dictionary learning, then no feature extraction needed
      features=extract_raw_frame_features(userDataStruct.Video(:,:,:,ii),userDataStruct.Video(:,:,:,ii-1),ii,patchSize);
    end 
    userDataStruct.TrainData=[userDataStruct.TrainData; features];
end
modelstruct=userDataStruct.modelstruct;
modelstruct.ImageSize=size(userDataStruct.Video);
modelstruct.modelno=modelno;     

%Train
strmodel=event_detector_learn(userDataStruct.TrainData,modelstruct.modelno,3:5,patchSize,modelstruct.ImageSize(1:2));

modelstruct.model=strmodel.model;
modelstruct.Xtr=strmodel.Xtr;
modelstruct.ytr=strmodel.ytr;
modelstruct.patchSize=strmodel.patchSize;
modelstruct.stackDepth=strmodel.stackDepth;
modelstruct.W=strmodel.W;
modelstruct.traindatamean=strmodel.traindatamean;
modelstruct.traindatastd=strmodel.traindatastd;
modelstruct.Dred=strmodel.Dred;
if modelno<=4
    modelstruct.trainErrorMean=strmodel.trainErrorMean;
    modelstruct.trainErrorStd=strmodel.trainErrorStd;
end

userDataStruct.modelstruct=modelstruct;

% Predict
if modelno<=4
     featuresprev=extract_frame_features(userDataStruct.Video(:,:,:,10),userDataStruct.Video(:,:,:,9),10,userDataStruct.modelstruct.patchSize);
else % if dictionary learning, then no feature extraction needed
     featuresprev=extract_raw_frame_features(userDataStruct.Video(:,:,:,10),userDataStruct.Video(:,:,:,9),10,userDataStruct.modelstruct.patchSize);
end
numTestFrames=0;

 for ii=11:134
    ii
    userDataStruct.currentImageIdx=ii;
    
    I=userDataStruct.Video(:,:,:,ii);
    
    if modelno<=4
         features=extract_frame_features(userDataStruct.Video(:,:,:,ii),userDataStruct.Video(:,:,:,ii-1),ii,userDataStruct.modelstruct.patchSize);
    else % if dictionary learning, then no feature extraction needed
         features=extract_raw_frame_features(userDataStruct.Video(:,:,:,ii),userDataStruct.Video(:,:,:,ii-1),ii,userDataStruct.modelstruct.patchSize);
    end
    % Predict!
    Iheat_ts_ii=event_detector_predict(userDataStruct.modelstruct,[featuresprev; features],ii);
    if modelno<=4
        %Iheat_ts_ii(Iheat_ts_ii<(userDataStruct.modelstruct.trainErrorMean+4*userDataStruct.modelstruct.trainErrorStd))=0;
    else
        Iheat_ts_ii=Iheat_ts_ii-modelstruct.model.maxTrainActivations;
    end
    %Iheat_ts_ii=floor(Iheat_ts_ii/max(Iheat_ts_ii(:))*255);
    
    I=userDataStruct.Video(:,:,:,ii);
    I(:,:,1)=Iheat_ts_ii;
    
    numTestFrames=numTestFrames+1;
    Iheat_ts(:,:,numTestFrames)=Iheat_ts_ii;
    
    featuresprev=features;
    
    testFrames=[testFrames; ii];
    
    %figure(1);imshow(I);
    %pause(0.1);
 end
 
 % save the results
 save(['/home/mkandemi/Documents/results/zeiss/video' num2str(vidno) '_results_model' num2str(modelno) '_ver2'],'Iheat_ts','patchSize','testFrames');
