function trainBayesian_sets(root_dir1, filelist1,root_dir2, filelist2, version)
    root_dir1=[root_dir1 '/'];
    label_list1=textread(filelist1,'%s','delimiter','\n');
    length1=size(label_list1,1);
    num1=length1/2;
    root_dir2=[root_dir2 '/'];
    label_list2=textread(filelist2,'%s','delimiter','\n');
    length2=size(label_list2,1);
    num2=length2/2;
    if(1)
        load('PCA.mat')
        feat_lenth=size(projectMat,2);
        featmat=zeros(num1+num2,feat_lenth);
        labelmat=zeros(num1+num2,1);
        fprintf('start load filelist1\n');
        for i=1:num1
            if(mod(i,5000)==0)
                fprintf('i = %d\n',i);
            end
            labelmat(i)=20000+str2num(label_list1{2*i});%20000避免两个数据库标签重复
            feat_mat=load([root_dir1 label_list1{2*i-1} '.crop_' version  '.jpg.mat']);
            pca_feat=(feat_mat.feat'-featMean)*projectMat;
            featmat(i,:)=pca_feat';
        end
        fprintf('start load filelist2\n');
        for i=1:num2
            if(mod(i,5000)==0)
                fprintf('i = %d\n',i);
            end
            labelmat(num1+i)=str2num(label_list2{2*i});
            feat_mat=load([root_dir2 label_list2{2*i-1} '.crop_' version  '.jpg.mat']);
            pca_feat=(feat_mat.feat'-featMean)*projectMat;
            featmat(i+num1,:)=pca_feat';
        end
        save(['test_cell_' version '.mat'],'labelmat','featmat');
    else
        load(['test_cell_' version '.mat']);
    end
    fprintf('start JointBayesian training\n')
    index=1;
%     for i=1:num
%         if((mod(i,15)>5)||(mod(i,15)==0))
%             featmat(index,:)=[];
%             labelmat(index)=[];
%         else
%             index=index+1;
%         end
%     end
% featmat(num/2:end,:)=[];
% labelmat(num/2:end,:)=[];
    [mappedX, mapping] = JointBayesian(featmat, labelmat);
    save('mapping.mat','mapping');
    
    
end