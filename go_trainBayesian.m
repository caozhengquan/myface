root_dir1='D:\����\CASIA-WebFace';
labellist1='D:\����\webfaceout\labellist_list.txt';
root_dir2='D:\����';
labellist2='D:\����\muct-master\all\out_neww\train_list.txt';
version='02';
%trainBayesian(root_dir1,labellist1,version)
trainBayesian(root_dir2,labellist2,version)
%trainBayesian_sets(root_dir1,labellist1,root_dir2,labellist2,version)
