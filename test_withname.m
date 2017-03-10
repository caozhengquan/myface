function test_withname(root_dir,testlabel_dir,version)
root_dir=[root_dir '/'];
label_list=textread(testlabel_dir,'%s');
length=size(label_list,1);
num=length/2;
if(1)
 load('PCA.mat')
 load('mapping.mat')
test_cell=cell(num,3);
for i=1:num
    test_cell{i,1}=[root_dir label_list{2*i-1}];
    test_cell{i,2}=label_list{2*i};
    feat_mat=load([root_dir label_list{2*i-1} '.crop_' version  '.jpg.mat']);
    pca_feat=(feat_mat.feat'-featMean)*projectMat;
    test_cell{i,3}=pca_feat';
    %test_cell{i,3}=feat_mat.feat;
end
save(['test_cell_' version '.mat'],'test_cell')
else
    load(['test_cell_' version '.mat'])
end

if(1)
score_inter=[];
score_inner=[];
%load('PCA.mat')
for i=1:num
    if(mod(i,1000)==0)
        i
    end
    for j=i+1:num
        res_i=test_cell{i,3};
        res_j=test_cell{j,3};
%         res_i=(res_i'-featMean)*projectMat;
%         res_j=(res_j'-featMean)*projectMat;
%         res_i=res_i';
%         res_j=res_j';
          diff=res_i-res_j;
           %score=norm(diff);
%score=res_i'*res_j/norm(res_i)/norm(res_j);
score=res_i' * mapping.A * res_i + res_j' * mapping.A * res_j - 2 * res_i' * mapping.G *res_j;
        if(strcmp(test_cell{i,2},test_cell{j,2}))
            score_inner=[score_inner;score];
        else
            score_inter=[score_inter;score];
        end
    end
end

save(['score_' version '.mat'],'score_inner','score_inter');
else
    load(['score_' version '.mat'])
end
x=zeros(1,100);
y=x;

min_s=min(score_inner);
max_s=max(score_inter);
 for i=1:100

% thr=10+0.4*i;
%   x(i)=size(find(score_inter<thr),1);
%   y(i)=size(find(score_inner>thr),1);
  %thr=0.5+0.008*i;
  thr=min_s+(max_s-min_s)*i/100;
  x(i)=size(find(score_inter>thr),1);
  y(i)=size(find(score_inner<thr),1);
end
x=x/size(score_inter,1);
y=y/size(score_inner,1);
save(['roc_' version '.mat'],'x','y');
plot(x,y)

caffe.reset_all();
end


