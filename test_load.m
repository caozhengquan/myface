
model = ['deploy.prototxt'];
weights = [model_path 'model.caffemodel'];
net = caffe.Net(model, weights, 'test');