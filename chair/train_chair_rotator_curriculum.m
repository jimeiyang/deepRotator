addpath(genpath('../../matlab'));
% prepare oversampled input
path_to_data  = 'data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
train_idx = ids<=500;
test_idx = ids>500;
images_train = images(:,:,:,train_idx);
[h,w,c,numtrains] = size(images_train);
ids_train = ids(train_idx);
phi_train = phi(train_idx);
theta_train = theta(train_idx);
ids_types = unique(ids_train);
phi_types = unique(phi_train);
theta_types = unique(theta_train);
batch_size = length(phi_types(:));

path_to_results = 'results/';
if ~isdir(path_to_results), mkdir(path_to_results); end


% train rnn model 
for numsteps = [2,4,8,16],
    % init caffe network (spews logging info)
    model_specs = sprintf('rnn_t%d_finetuning',numsteps);
    use_gpu = true;
    model_file = sprintf('%s.prototxt', model_specs);
    solver_file = sprintf('%s_solver.prototxt', model_specs);
    param = struct('base_lr', 0.000001, 'stepsize', 1000000, 'weight_decay', 0.001, 'solver_type', 3);
    make_solver_file(solver_file, model_file, param);
    init_matcaffe(solver_file, use_gpu, 1);

    % load pretrained layer weights
    model = caffe('get_weights');
    if numsteps == 2,
        model_pretrained = load('models/base_model_iter0400.mat');
    else
        model_pretrained = load(sprintf('models/rnn_t%d_finetuning_model_iter0050.mat', numsteps/2));
    end
    mapping = [1:7;1:7];
    for i = 1:numsteps, mapping = [mapping, [(8:18)+(i-1)*11; 8:18]]; end
    for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
    caffe('set_weights', model);

    fid_train = fopen([path_to_results sprintf('%s_train_errors.txt',model_specs)],'w');
    for n = 1:50
        loss_train_image = 0;
        tic;
        fprintf('%s -- processing the %dth iteration.\n', model_specs, n);
        m = length(ids_types(:))*length(theta_types(:))*length(phi_types(:))*2;
        for cc = ids_types(:)', % chair instance
            for tt = theta_types(:)', % elevation
                batch_idx = find(ids_train==cc & theta_train==tt);
                images_batch = images_train(:,:,:,batch_idx);
                images_batch = single(permute(images_batch,[2,1,3,4]))/255;    
                masks_batch = single(mean(images_batch,3)>0);
                phi_batch = phi_train(batch_idx);
                [phi_batch, order] = sort(phi_batch, 'ascend');
                images_batch = images_batch(:,:,:,order);
                masks_batch = masks_batch(:,:,:,order);
                rnd_idx = randperm(batch_size);
                rnd_idx = rnd_idx(1:4);
                for ii = rnd_idx(:)',
                    for label = [1 3],
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
                        [loss_image, delta_image] = loss_euclidean_grad(recons_image, images_out);
                        [loss_mask, delta_mask] = loss_euclidean_grad(recons_mask, masks_out);
                        %fprintf('Done with delta\n');
                        caffe('backward', {delta_image; delta_mask});
                        caffe('update');
                        %fprintf('Done with update\n');
                        loss_train_image = loss_train_image + (10*loss_image+loss_mask)/numsteps;
                    end
                end
            end
        end      
        loss_train_image = loss_train_image / m;
        fprintf(sprintf('%s -- training losses are %f for images in %f seconds.\n', model_specs, loss_train_image, toc));
        fprintf(fid_train, '%d %f\n', n, loss_train_image); 
    end
    fclose(fid_train);
    weights = caffe('get_weights');
    save(sprintf(['models/%s_model_iter%04d.mat'], model_specs, n), 'weights');
end

