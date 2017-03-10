function trainPCA_sets(root_dir1,filelist1,root_dir2, filelist2, version)
    filelist1_=textread(filelist1,'%s','delimiter','\n');
    num_face1=size(filelist1_,1);
    feat_1_p=[root_dir1 '/' filelist1_{1}  '.crop_'  version  '.jpg.mat'];
    feat_1c=load(feat_1_p);
    feat_1=feat_1c.feat;
    filelist2_=textread(filelist2,'%s','delimiter','\n');
    
    num_face2=size(filelist2_,1);

    feat_vec=zeros(num_face1+num_face2,size(feat_1,1));
    for i=1:num_face1
        img_name=filelist1_{i};
        feat_name=[root_dir1 '/' img_name  '.crop_'  version  '.jpg.mat'];
         feat_c=load(feat_name);
         feat=feat_c.feat;
         feat_vec(i,:)=feat';
    end
    for i=1:num_face2
        img_name=filelist2_{i};
        feat_name=[root_dir2 '/' img_name  '.crop_'  version  '.jpg.mat'];
         feat_c=load(feat_name);
         feat=feat_c.feat;
         feat_vec(i,:)=feat';
    end
    [projectMat featMean]=myPCA(feat_vec,100);
    save('PCA.mat','projectMat','featMean');
end