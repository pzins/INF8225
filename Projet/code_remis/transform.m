load wiki.mat
path=wiki.full_path();
gender=wiki.gender()+1;
[age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob); 

N=length(path);

faceDetector = vision.CascadeObjectDetector('FrontalFaceLBP');
DataSetX=zeros(N,64*64*3);
DataSetGender=zeros(N,1);
DataSetAge=zeros(N,1);

disp('Début de la lecture des images')

for i=1:N
  
    % Check si le fichier peut être lu. Si NON il passe +1
    while (((exist(strjoin(path(1,i))))==0||(age(i)<20)||(age(i)>40)||(isnan(gender(i))==1))&&(i<N))
        i=i+1;
    end
    
    disp(['itération ', num2str(i), ' sur ', num2str(length(path)), ' ... ']);
    
    I=imread(strjoin(path(1,i)));
        if(size(I,3)==1)
        I=I(:,:,[1 1 1]);
        end
        
    bboxes = step(faceDetector, I);
    
        if(isempty(bboxes)==0)
            
        bboxes(1,:)=[bboxes(1,1)*0.5 bboxes(1,2)*0.5 bboxes(1,3)*1.5 bboxes(1,4)*1.5];
        I=imcrop(I,bboxes(1,:));
        B = imresize(I,[64 64],'bicubic');
        C=reshape(B,[1,numel(B)]);
        DataSetX(i,:)=C;
        DataSetGender(i,1)=gender(1,i);
        DataSetAge(i,1)=age(1,i);
        end
        
end

DataSetAge(all(DataSetAge==0,2),:)=[];
DataSetX(all(DataSetX==0,2),:)=[];
DataSetGender(all(DataSetGender==0,2),:)=[];
DataSetGender=DataSetGender-1;

disp('Pre-processing done')
