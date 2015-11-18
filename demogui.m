function varargout = demogui(varargin)
% DEMOGUI MATLAB code for demogui.fig
%      DEMOGUI, by itself, creates a new DEMOGUI or raises the existing
%      singleton*.
%
%      H = DEMOGUI returns the handle to a new DEMOGUI or the handle to
%      the existing singleton*.
%
%      DEMOGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DEMOGUI.M with the given input arguments.
%
%      DEMOGUI('Property','Value',...) creates a new DEMOGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before demogui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to demogui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to collectdatabutton (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help demogui

% Last Modified by GUIDE v2.5 27-May-2014 12:55:45

% Begin initialization code - DO NOT EDIT

%addpath(genpath('../vlfeat'));
%collectdatabutton('vl_setup');   

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @demogui_OpeningFcn, ...
                   'gui_OutputFcn',  @demogui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before demogui is made visible.
function demogui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to demogui (see VARARGIN)

% Choose default command line output for demogui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

%guidata(hObject,I);
%guidata(hObject,Iprevious);

% UIWAIT makes demogui wait for user response (see UIRESUME)
%uiwait(handles.ImagePanel);

Istart=zeros(600,800,3);
handles.ImagePanel;imshow(Istart);
 
userDataStruct.currentImageIdx=2;
userDataStruct.TrainData=[];
userDataStruct.modelstruct=[];
set(handles.collectdatabutton,'UserData',userDataStruct);


% --- Outputs from this function are returned to the command line.
function varargout = demogui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton_run.
function collectdatabutton_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_run (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

  isPushed=get(hObject,'String');
  userDataStruct=get(hObject,'UserData');
  currentImageIdx=userDataStruct.currentImageIdx; 
  
  NumFrames=size(userDataStruct.Video,4);
  
  if isequal(isPushed,'Collect Data')
      set(hObject,'String','Stop');
  else
      set(hObject,'String','Collect Data');
  end

 for ii=currentImageIdx:134
    if isequal(get(hObject,'String'),'Collect Data')
        % train and stop
               
        % Disable the button!
        set(handles.collectdatabutton,'Enable','off');
        set(handles.trainbutton,'Enable','on');
        
        break;
    end
    
    userDataStruct.currentImageIdx=ii;
    
    set(handles.frametext,'String',sprintf('Frame: %d/%d\n',ii,NumFrames));
        
    set(hObject,'UserData',userDataStruct);        
    handles.ImagePanel; imshow(userDataStruct.Video(:,:,:,ii));
    pause(1);
 end

% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename filepath] = uigetfile('*.avi');
filename=[filepath filename];
if ~isequal(filename, 0)
  %  open(file);
  fprintf('%s\n',filename);
  
   vid=VideoReader(filename);
   
   N=vid.NumberOfFrames;
      
   I=read(vid,[1 N]);
   
   userDataStruct=get(handles.collectdatabutton,'UserData');   
   userDataStruct.Video=I;
   userDataStruct.modelstruct.ImageSize=size(I);
   set(handles.collectdatabutton,'UserData',userDataStruct);
   
   set(handles.frametext,'String',sprintf('Frame: 1/%d\n',N));
   set(handles.collectdatabutton,'Enable','on');   
   
   handles.ImagePanel=I(:,:,:,1); imshow(handles.ImagePanel);


end

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.ImagePanel)

% --------------------------------------------------------------------trainbutton
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.ImagePanel,'Name') '?'],...
                     ['Close ' get(handles.ImagePanel,'Name') '...'],...
                     'Yes','No','Yes');
if strcmp(selection,'No')
    return;
end

delete(handles.ImagePanel)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
     set(hObject,'BackgroundColor','white');
end

%% Do Training
function trainbutton_Callback(hObject, eventdata, handles)
% hObject    handle to trainbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.collectdatabutton,'Enable','off');

userDataStruct=get(handles.collectdatabutton,'UserData');

patchSize=str2double(get(handles.patchsizeedit,'String'));

modelno=get(handles.modellist,'Value');

for ii=2:userDataStruct.currentImageIdx
    fprintf('Frame: %d/%d is being processed\n',ii,userDataStruct.currentImageIdx);
    if modelno <= 4
      features=extract_frame_features(userDataStruct.Video(:,:,:,ii),userDataStruct.Video(:,:,:,ii-1),ii,patchSize);   
    else % if dictionary learning, then no feature extraction needed
      features=extract_raw_frame_features(userDataStruct.Video(:,:,:,ii),ii,patchSize);           
    end 
    userDataStruct.TrainData=[userDataStruct.TrainData; features];
end

modelstruct=userDataStruct.modelstruct;

ImageSize=size(userDataStruct.Video);

modelstruct.modelno=modelno;       
strmodel=event_detector_learn(userDataStruct.TrainData,modelstruct.modelno,3:userDataStruct.currentImageIdx,patchSize,ImageSize(1:2));

modelstruct.model=strmodel.model;
modelstruct.Xtr=strmodel.Xtr;
modelstruct.ytr=strmodel.ytr;
modelstruct.patchSize=strmodel.patchSize;
modelstruct.stackDepth=strmodel.stackDepth;
modelstruct.W=strmodel.W;
modelstruct.traindatamean=strmodel.traindatamean;
modelstruct.traindatastd=strmodel.traindatastd;
modelstruct.Dred=strmodel.Dred;
modelstruct.trainErrorMean=strmodel.trainErrorMean;
modelstruct.trainErrorStd=strmodel.trainErrorStd;

userDataStruct.modelstruct=modelstruct;

set(handles.collectdatabutton,'UserData',userDataStruct);
set(handles.trainbutton,'Enable','off');
set(handles.predictbutton,'Enable','on');

%% Do prediction!
function predictbutton_Callback(hObject, eventdata, handles)

  set(handles.predictbutton,'Enable','off');
  userDataStruct=get(handles.collectdatabutton,'UserData');
  currentImageIdx=userDataStruct.currentImageIdx; 
  
  featuresprev=userDataStruct.TrainData;

 for ii=currentImageIdx+1:134
    
    userDataStruct.currentImageIdx=ii;
    
    I=userDataStruct.Video(:,:,:,ii);
    
    if userDataStruct.modelstruct.modelno<=4
         features=extract_frame_features(userDataStruct.Video(:,:,:,ii),userDataStruct.Video(:,:,:,ii-1),ii,userDataStruct.modelstruct.patchSize);       
    else % if dictionary learning, then no feature extraction needed
         features=extract_raw_frame_features(userDataStruct.Video(:,:,:,ii),ii,userDataStruct.modelstruct.patchSize);   
    end
    % Predict!
    Iheat_ts=event_detector_predict(userDataStruct.modelstruct,[featuresprev; features],ii);
    if userDataStruct.modelstruct.modelno<=4    
        Iheat_ts(Iheat_ts<(userDataStruct.modelstruct.trainErrorMean+4*userDataStruct.modelstruct.trainErrorStd))=0;
    end
    Iheat_ts=floor(Iheat_ts/max(Iheat_ts(:))*255);    
    
    I(:,:,1)=Iheat_ts;
    
    set(handles.frametext,'String',sprintf('Frame: %d/%d\n',ii,size(userDataStruct.Video,4)));
    
    % Draw predictions!
    set(hObject,'UserData',userDataStruct);        
    handles.ImagePanel; imshow(I);
    pause(0.3);
    
    
    featuresprev=features;
 end


% --- Executes on selection change in modellist.
function modellist_Callback(hObject, eventdata, handles)
% hObject    handle to modellist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns modellist contents as cell array
%        contents{get(hObject,'Value')} returns selected item from modellist

val=get(hObject,'Value')

% --- Executes during object creation, after setting all properties.
function modellist_CreateFcn(hObject, eventdata, handles)
% hObject    handle to modellist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function patchsizeedit_Callback(hObject, eventdata, handles)
% hObject    handle to patchsizeedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of patchsizeedit as text
%        str2double(get(hObject,'String')) returns contents of patchsizeedit as a double


% --- Executes during object creation, after setting all properties.
function patchsizeedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to patchsizeedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
