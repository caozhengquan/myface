function trainBayesian(root_dir, filelist, version)
    root_dir=[root_dir '/'];
    label_list=textread(filelist,'%s','delimiter','\n');
    length=size(label_list,1);
    num=length/2;
    if(1)
        load('PCA.mat')
        feat_lenth=size(projectMat,2);
        featmat=zeros(num,feat_lenth);
        labelmat=zeros(num,1);
        for i=1:num
            if(mod(i,5000)==0)
                fprintf('i = %d\n',i);
            end
            labelmat(i)=str2num(label_list{2*i});
            feat_mat=load([root_dir label_list{2*i-1} '.crop_' version  '.jpg.mat']);
            pca_feat=(feat_mat.feat'-featMean)*projectMat;
            featmat(i,:)=pca_feat';
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
    %[mappedX, mapping] = JointBayesian(featmat, labelmat);
     [mappedX, mapping] = TransferBayesian(featmat, labelmat);
    save('mapping.mat','mapping');
    
    
end