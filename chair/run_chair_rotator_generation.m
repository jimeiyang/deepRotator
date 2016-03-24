addpath(genpath('../../matlab'));
% prepare oversampled input
path_to_data  = 'data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = ((ids>500) & (ids<=1000));
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
numsteps = 16;
model_specs = sprintf('rnn_t%d_finetuning',numsteps);
use_gpu = true;
model_file = sprintf('models/%s.prototxt', model_specs);
solver_file = sprintf('models/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.000001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
init_matcaffe(solver_file, use_gpu, 1);

% load layer weights
rnn = 16;
model = caffe('get_weights');
model_pretrained = load(sprintf('models/rnn_t%d_finetuning_model_iter0050.mat',16));
mapping = [1:7;1:7]; 
for i = 1:numsteps, mapping = [mapping, [(8:18)+(i-1)*11; 8:18]]; end
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);

path_to_results = 'results/';
if ~isdir(path_to_results), mkdir(path_to_results); end
for cc = ids_types(:)', % chair instance
    tic
    for tt = theta_types(:)', % elevation
        batch_idx = find(ids_test==cc & theta_test==tt);
        images_batch = images_test(:,:,:,batch_idx);
        images_batch = single(permute(images_batch,[2,1,3,4]))/255;    
        masks_batch = single(mean(images_batch,3)>0);
        phi_batch = phi_test(batch_idx);
        [phi_batch, order] = sort(phi_batch, 'ascend');
        images_batch = images_batch(:,:,:,order);
        masks_batch = masks_batch(:,:,:,order);
	output = zeros([w,h,c*(rnn+1),batch_size,2], 'single');
	output_mask = zeros([w,h,(rnn+1),batch_size,2], 'single');
        for ii = 1:batch_size, % azimuth
            jj = 1;
            for label = [1 3], % direction
                action = zeros(1,1,3,1,'single'); 
                action(1,1,label,1) = 1;
                input = cell(1+numsteps,1);
                input{1} = images_batch(:,:,:,ii);
                if label == 1, idx_rot = [ii+1:batch_size,1:ii]; end
                if label == 3, idx_rot = [fliplr(1:ii-1),fliplr(ii:batch_size)]; end
                images_out = images_batch(:,:,:,idx_rot);
                images_out = reshape(images_out(:,:,:,1:numsteps), [w,h,3*numsteps,1]);
                masks_out = masks_batch(:,:,:,idx_rot);
                masks_out = reshape(masks_out(:,:,:,1:numsteps), [w,h,1*numsteps,1]); 
                for ss = 1:numsteps, input{ss+1} = action; end
                results = caffe('forward', input);
                %fprintf('Done with forward pass.\n');
                recons_image = results{1};
                recons_mask = results{2};
                output(:,:,1:3,ii,jj) = images_batch(:,:,:,ii);        
		output(:,:,4:c*(rnn+1),ii,jj) = recons_image;
                output_mask(:,:,1,ii,jj) = masks_batch(:,:,:,ii);        
		output_mask(:,:,2:rnn+1,ii,jj) = recons_mask;
                jj = jj + 1;
%                visualize_predictions([path_to_results sprintf('preds%d_inst%d_ele%d_azi%d_act%d_t%d.avi',rnn,cc,tt,ii,label,numsteps)], input{1}, recons_image);
            end
        end
%        visualize_predictions_batch([path_to_results sprintf('preds%d_inst%d_ele%d_t%d.avi',rnn,cc,tt,numsteps)], output);
       save([path_to_results sprintf('preds%d_inst%d_ele%d_t%d.mat',rnn,cc,tt,numsteps)], 'output', 'output_mask');
    end
    fprintf('rotating chair %d in %f seconds\n', cc, toc);
end      

