function test(root_dir,filelist,pairlist)
clc;
version='02';
show_inner=0;
show_inter=0;
multi_cos=0;
show_thre=-0.45;
nume_inner=0;
nume_inter=0;
filelist_=textread(filelist,'%s','delimiter','\n');
num_face=size(filelist_);

%addpath('/home/zq/caffe/matlab');
if 1
caffe.reset_all();

% load face model and creat network
caffe.set_device(0);
 caffe.set_mode_gpu();
% model = './deploy_res.prototxt';
% weights = './face_model_res.caffemodel';
% model = './deploy_new.prototxt';
% weights = './face_model_new.caffemodel';
model = './light_deploy.prototxt';
weights = './light.caffemodel';
net = caffe.Net(model, weights, 'test');

for i=1:num_face
    if(mod(i,500)==0)
        i
    end
    filename=filelist_{i};
    crop_filename=[root_dir '/' filename  '.crop_'  version  '.jpg'];
    cropImg=prepare_image(crop_filename);
   %tic
    res=net.forward(cropImg);
    %toc
    %feat=[res{1}(:,1) ;res{1}(:,2)];
    feat=res{1};
    save([crop_filename '.mat'],'feat');
end
caffe.reset_all();
end


load('PCA.mat')
score_inter=[];
score_inner=[];
fid=fopen(pairlist,'r');
fgets(fid)

for i=1:10
    i
    for j=1:300
        a=fgets(fid);
        S = regexp(a,'\s+', 'split');
        index1=sprintf('%04d',str2num(S{2}));
        index2=sprintf('%04d',str2num(S{3}));
        path1=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg.mat'];
        path2=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg.crop_' version  '.jpg.mat'];
         path1cp=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg'];
        path2cp=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg.crop_' version  '.jpg'];
           path1p=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg'];
        path2p=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg'];
        data1=load(path1);
        data2=load(path2);
        feat1=data1.feat;
        feat2=data2.feat;
        feat1=(feat1'-featMean)*projectMat;
        feat2=(feat2'-featMean)*projectMat;
    feat1=feat1';
    feat2=feat2';
        diff=feat1-feat2;
        score=norm(diff);
 score=-feat1'*feat2/norm(feat1)/norm(feat2);
 if(multi_cos)       
 feat1_=feat1-0.5*ones(size(feat1));
          feat2_=feat2-0.5*ones(size(feat2));
         
          score=score-feat1_'*feat2_/norm(feat1_)/norm(feat2_);
 end
         if(score/(multi_cos+1)>show_thre)
             if(show_inner)
                    a=imread(path1cp);
                    b=imread(path2cp);
                    c=[a b];
                    imshow(c);
                    pause(0.2);
             end
             nume_inner=nume_inner+1;
         end
%          feat1__=feat1-1*ones(size(feat1));
%          feat2__=feat2-1*ones(size(feat2));
%          score=score-feat1__'*feat2__/norm(feat1__)/norm(feat2__);
         
         score_inner=[score_inner;score/(1+multi_cos)];
    end
    for j=1:300
         a=fgets(fid);
        S = regexp(a,'\s+', 'split');
        index1=sprintf('%04d',str2num(S{2}));
        index2=sprintf('%04d',str2num(S{4}));
        path1=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg.mat'];
        path2=[root_dir '/lfw/' S{3} '/' S{3} '_' index2 '.jpg.crop_' version  '.jpg.mat'];
        
         path1cp=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg'];
        path2cp=[root_dir '/lfw/' S{3} '/' S{3} '_' index2 '.jpg.crop_' version  '.jpg'];
           path1p=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg'];
        path2p=[root_dir '/lfw/' S{3} '/' S{3} '_' index2 '.jpg'];
        
        data1=load(path1);
        data2=load(path2);
        feat1=data1.feat;
        feat2=data2.feat;
         feat1=(feat1'-featMean)*projectMat;
        feat2=(feat2'-featMean)*projectMat;
    feat1=feat1';
    feat2=feat2';
         diff=feat1-feat2;
         score=norm(diff);
          score=-feat1'*feat2/norm(feat1)/norm(feat2);
        if(multi_cos)
            feat1_=feat1-0.5*ones(size(feat1));
            feat2_=feat2-0.5*ones(size(feat2));
       
            score=score-feat1_'*feat2_/norm(feat1_)/norm(feat2_);
        end
%            feat1__=feat1-1*ones(size(feat1));
%          feat2__=feat2-1*ones(size(feat2));
%          score=score-feat1__'*feat2__/norm(feat1__)/norm(feat2__);
  if(score/(1+multi_cos)<show_thre)
            if(show_inter)
                    a=imread(path1cp);
                    b=imread(path2cp);
                    c=[a b];
                    imshow(c);
                    pause(0.2);
            end
             nume_inter=nume_inter+1;
   end
         score_inter=[score_inter;score/(1+multi_cos)];
    end
 
end
fprintf('nume_inner is : %d\n',nume_inner);
fprintf('nume_inter is : %d\n',nume_inter);
save(['score_' version '.mat'],'score_inner','score_inter');
sum_acc=0;
for i=1:10
    score_i_train=score_inter;
    score_i_train( ( (i-1)*300+1 ) : (i*300) )=[];
    score_i_test=score_inter( ( (i-1)*300+1 ) : (i*300) );
    score_inn_train=score_inner;
    score_inn_train( ( (i-1)*300+1 ) : (i*300) )=[];
    score_inn_test=score_inner( ( (i-1)*300+1 ) : (i*300) );
    x=zeros(1,1000);
    y=x;
    z=x;
    a=18;
    b=0.02;
    a=-0.8;
     b=0.001;
     for j=1:1000

        %thr=40+0.08*i;
        %thr=0.2+0.0008*i;
        thr=a+b*j;
         x(j)=size(find(score_i_train<thr),1);
         y(j)=size(find(score_inn_train>thr),1);
        z(j)=x(j)+y(j);
        %   thr=0.5+0.008*i;
        %   x(i)=size(find(score_inter>thr),1);
        %   y(i)=size(find(score_inner<thr),1);
     end
  min_z=min(z);
   min_z_index=min(find(z==min_z));
   thr_train=a+b*min_z_index;
   
   acc_r=1-(size(find(score_i_test<thr_train),1)+size(find(score_inn_test>thr_train),1))/600;
    sum_acc=sum_acc+acc_r;
end
acc=sum_acc/10;
fprintf('acc is %d\n',acc);

x=zeros(1,1000);
y=x;
z=x;
 for i=1:1000

%thr=40+0.08*i;
%thr=0.2+0.0008*i;
thr=a+b*i;
  x(i)=size(find(score_inter<thr),1);
  y(i)=size(find(score_inner>thr),1);
  z(i)=x(i)+y(i);
%   thr=0.5+0.008*i;
%   x(i)=size(find(score_inter>thr),1);
%   y(i)=size(find(score_inner<thr),1);
end
x=x/size(score_inter,1);
y=y/size(score_inner,1);
save(['roc_' version '.mat'],'x','y');
plot(x,y)
false_n=min(z);
accu_rate=1-false_n/6000;
fprintf('accu_rate is %s\n',accu_rate);

end

function crops_data = prepare_image(im_name)
cropImg = imread(im_name);
% transform image, obtaining the original face and the horizontally flipped one
if size(cropImg, 3) < 3
   cropImg(:,:,2) = cropImg(:,:,1);
   cropImg(:,:,3) = cropImg(:,:,1);
end
cropImg = single(cropImg);
cropImg = (cropImg - 127.5)/128;
cropImg = permute(cropImg, [2,1,3]);
cropImg = cropImg(:,:,[3,2,1]);
%cropImg_=flipud(cropImg);
%imshow(cropImg_)
% output=zeros(size(cropImg,1),size(cropImg,2),size(cropImg,3),2);
% output(:,:,:,1)=cropImg;
% output(:,:,:,2)=cropImg_;
output=cropImg;
crops_data={output};
%crops_data={[[cropImg] [cropImg]]};
end
