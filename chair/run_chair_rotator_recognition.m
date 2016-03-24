% add path
addpath(genpath('../../matlab'));
% prepare oversampled input
path_to_data  = 'data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = ids > 500;
images_test = images(:,:,:,test_idx);
[h,w,c,numtests] = size(images_test);
ids_test = ids(test_idx);
phi_test = phi(test_idx);
theta_test = theta(test_idx);
ids_types = unique(ids_test);
phi_types = unique(phi_test);
theta_types = unique(theta_test);
batch_size = length(phi_types(:));


% extract features
numsteps = 16;
feats_test = cell(1,length(ids_types(:))*length(theta_types(:)));
path_to_results = 'feas/';
m = 1;
for cc = ids_types(:)', % chair instance
%	tic
    for tt = theta_types(:)', % elevation
        data = load([path_to_results sprintf('feas_t%d_inst%d_ele%d.mat',numsteps,cc,tt)]);
        feats_test{m} = squeeze(data.results{2});
        m = m + 1;
    end
%	fprintf('--class %d in %f seconds\n',cc, toc);
end      
feats_test = cell2mat(feats_test);
feats_test = feats_test ./ repmat(sqrt(sum(feats_test.^2,1)),512,1);

% gallery and probe split
acc_test = zeros(31,15);
for i = 1:31,

    select_phi = phi_types(i);
    ind_gallery = (phi_test == select_phi);
    ind_probe = (~ind_gallery);		

    % gallery/probe split
    feats_gallery = feats_test(:,ind_gallery);
    feats_probe = feats_test(:,ind_probe);
    ids_gallery = ids_test(ind_gallery);
    ids_probe = ids_test(ind_probe);
    phi_gallery = phi_test(ind_gallery);
    phi_probe = phi_test(ind_probe);

    % fix phi degree
    delta = 360/31;
    phi_gallery = delta*round(phi_gallery/delta);
    phi_probe = delta*round(phi_probe/delta);
    select_phi = delta*round(select_phi/delta);
        diff = abs(phi_probe(:) - select_phi);
    diff = (diff>180).*(360-diff) + (diff<=180).*diff;
    diff = round(diff);

    % similarity
    dist = EuDist2(feats_gallery', feats_probe', 1); 
    [~,matches] = min(dist,[],1);
    ids_pred = ids_gallery(matches);

    % breakdown
    diff_types = unique(diff(:));
    acc = [];
    for d = diff_types(:)',
        ids_pred_sub = ids_pred(diff==d);
        ids_probe_sub = ids_probe(diff==d);
        acc = [acc; sum(ids_pred_sub==ids_probe_sub)/length(ids_probe_sub)];
    end
    acc_test(i,:) = acc(:)';
end
acc_test(:)'
save(sprintf('results/acc_test_t%d_posefeat.mat', numsteps), 'acc_test');