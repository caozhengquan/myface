function test(root_dir,filelist,pairlist)
clc;
version='01';
filelist_=textread(filelist,'%s','delimiter','\n');
num_face=size(filelist_);

%addpath('/home/zq/caffe/matlab');
if  1
caffe.reset_all();

% load face model and creat network
caffe.set_device(0);
caffe.set_mode_gpu();
model = './lr_deploy_res.prototxt';
weights = './face_model_res.caffemodel';
net = caffe.Net(model, weights, 'test');

for i=1:num_face
    if(mod(i,500)==0)
        i
    end
    filename=filelist_{i};
    crop_filename=[root_dir '/' filename  '.crop_'  version  '.jpg'];
    cropImg=prepare_image(crop_filename);
    res=net.forward(cropImg);
    feat=[res{1}(:,1) ;res{1}(:,2)];
    save([crop_filename 'lr.mat'],'feat');
end
caffe.reset_all();
end



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
        path1=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpglr.mat'];
        path2=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg.crop_' version  '.jpglr.mat'];
        
        data1=load(path1);
        data2=load(path2);
        feat1=data1.feat;
        feat2=data2.feat;
        diff=feat1-feat2;
        score=norm(diff);
         score_inner=[score_inner;score];
    end
    for j=1:300
         a=fgets(fid);
        S = regexp(a,'\s+', 'split');
        index1=sprintf('%04d',str2num(S{2}));
        index2=sprintf('%04d',str2num(S{4}));
        path1=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpglr.mat'];
        path2=[root_dir '/lfw/' S{3} '/' S{3} '_' index2 '.jpg.crop_' version  '.jpglr.mat'];
        
        data1=load(path1);
        data2=load(path2);
        feat1=data1.feat;
        feat2=data2.feat;
         diff=feat1-feat2;
         score=norm(diff);
         score_inter=[score_inter;score];
    end
 
end

save(['score_' version '.mat'],'score_inner','score_inter');

x=zeros(1,1000);
y=x;
z=x;
 for i=1:1000

%thr=40+0.08*i;
%thr=0.2+0.0008*i;
thr=20+0.04*i;
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
cropImg_=flipud(cropImg);
%imshow(cropImg_)
output=zeros(size(cropImg,1),size(cropImg,2),size(cropImg,3),2);
output(:,:,:,1)=cropImg;
output(:,:,:,2)=cropImg_;
crops_data={output};
%crops_data={[[cropImg] [cropImg]]};
end
