function trainPCA(root_dir, filelist, version)
    filelist_=textread(filelist,'%s','delimiter','\n');
    num_face=size(filelist_,1);
    feat_1_p=[root_dir '/' filelist_{1}  '.crop_'  version  '.jpg.mat'];
    feat_1c=load(feat_1_p);
    feat_1=feat_1c.feat;
    feat_vec=zeros(num_face,size(feat_1,1));
    for i=1:num_face
        img_name=filelist_{i};
        feat_name=[root_dir '/' img_name  '.crop_'  version  '.jpg.mat'];
         feat_c=load(feat_name);
         feat=feat_c.feat;
         feat_vec(i,:)=feat';
    end
    [projectMat featMean]=myPCA(feat_vec,100);
    save('PCA.mat','projectMat','featMean');
end