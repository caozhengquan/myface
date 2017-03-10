root_dir1='D:\数据\CASIA-WebFace';
labellist1='D:\数据\webfaceout\labellist_list.txt';
root_dir2='D:\数据';
labellist2='D:\数据\muct-master\all\out_neww\train_list.txt';
version='02';
%trainBayesian(root_dir1,labellist1,version)
trainBayesian(root_dir2,labellist2,version)
%trainBayesian_sets(root_dir1,labellist1,root_dir2,labellist2,version)
