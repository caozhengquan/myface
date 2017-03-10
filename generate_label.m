function generate_label(filelist_dir,labellist_dir)
    filelist=textread(filelist_dir,'%s');
    fid=fopen(labellist_dir,'w');
    num=size(filelist,1);
   % filename_tmp='';
    for i=1:num
    	filename=filelist{i};
        pos=find(filename=='\');
        %label_name=filename(pos(end)+2:pos(end)+4);
        label_name=filename(1:pos(1)-1);
        label_id=str2num(label_name);
        fprintf(fid,'%s\n',filename);
        fprintf(fid,'%d\n',label_id);
    end
    fclose(fid);
    disp('generate label_list complete');
end