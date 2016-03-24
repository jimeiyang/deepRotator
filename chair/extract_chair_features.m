addpath(genpath('../../matlab'));
% prepare oversampled input
path_to_data  = 'data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = (ids>500) || (ids<520);
images_test = images(:,:,:,test_idx);
[h,w,c,numtests] = size(images_test);
ids_test = ids(test_idx);
phi_test = phi(test_idx);
theta_test = theta(test_idx);
ids_types = unique(ids_test);
phi_types = unique(phi_test);
theta_types = unique(theta_test);
batch_size = length(phi_types(:));


% init caffe network (spews logging info)
model_specs = 'encoder';
use_gpu = true;
model_file = sprintf('models/%s.prototxt', model_specs);
solver_file = sprintf('models/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.000001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
init_matcaffe(solver_file, use_gpu, 3);

% load layer weights
rnn = 16;
model = caffe('get_weights');
model_pretrained = load(sprintf('models/rnn_t%d_finetuning_model_iter0050.mat',rnn));
mapping = [1:7;1:7]; 
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);

path_to_fea = 'feas/';
if ~isdir(path_to_fea), mkdir(path_to_fea); end
for cc = ids_types(:)', % chair instance
	tic
	for tt = theta_types(:)', % elevation
	    batch_idx = find(ids_test==cc & theta_test==tt);
	    images_batch = images_test(:,:,:,batch_idx);
	    images_batch = single(permute(images_batch,[2,1,3,4]))/255;    
	    phi_batch = phi_test(batch_idx);
	    [phi_batch, order] = sort(phi_batch, 'ascend');
	    images_batch = images_batch(:,:,:,order);
	    results = caffe('forward', {images_batch});
	    save([path_to_fea sprintf('feas_t%d_inst%d_ele%d.mat',rnn,cc,tt)], 'images_batch', 'results');
	end
	fprintf('--class %d in %f seconds\n',cc, toc);
end      
