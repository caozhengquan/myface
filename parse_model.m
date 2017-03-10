model='light_deploy.prototxt';
weights='light_0.003.caffemodel';
caffe.reset_all;
% create net and load weights
net = caffe.Net(model, weights, 'test'); 
num_layers=size(net.layer_names,1)
layer_names_vec=net.layer_names;
for i=1:num_layers
    layer_name=layer_names_vec{i};
    layer=net.layers(layer_name);
    layer_type=layer.type;
    if(strcmp(layer_type,'Convolution'))
       
        layer_file=['./weight/' layer_name '.dat'];
        w=layer.params(1).get_data();
        
        b=layer.params(2).get_data();
        %save(layer_file,'w','b');
        w_size=size(w);
        b_size=size(b);
        %open file
        fid=fopen(layer_file,'wb');
        fwrite(fid,w_size(1),'int');
        fwrite(fid,w_size(3),'int');
        fwrite(fid,w_size(4),'int');
        %write w
        for out_c=1:w_size(4)
            for in_c=1:w_size(3)
                fwrite(fid,w(:,:,in_c,out_c),'single');
            end
        end
        %write b
        fwrite(fid,b,'single');
        %close file
        fclose(fid);
    elseif(strcmp(layer_type,'InnerProduct'))
        layer_file=['./weight/' layer_name '.dat'];
        w=layer.params(1).get_data(); 
        b=layer.params(2).get_data();
        %save(layer_file,'w','b');
        w_size=size(w);
        b_size=size(b);
         %open file
        fid=fopen(layer_file,'wb');
        
        fwrite(fid,w_size(1),'int');
        fwrite(fid,w_size(2),'int');
        %write w
        fwrite(fid,w,'single');
        %write b
        fwrite(fid,b,'single');
        
        fclose(fid)
    end

end
