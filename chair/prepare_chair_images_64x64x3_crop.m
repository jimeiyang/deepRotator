% prepare batches for chairs
tic
load('data/chairs/rendered_chairs/all_chair_names.mat');
load('examples/chairs/reduced_set1.mat');
folder_names = folder_names(selected);
imfiles = cell(length(folder_names)*length(instance_names),1);
mskfiles = cell(length(folder_names)*length(instance_names),1);
ids = zeros(length(folder_names)*length(instance_names),1);
theta = zeros(1,length(folder_names)*length(instance_names));
phi = zeros(1,length(folder_names)*length(instance_names));
for i=1:length(folder_names)
    for j=1:length(instance_names)
        imfiles{(i-1)*length(instance_names)+j} = ['../caffe-master/data/chairs/cropped_chairs/' folder_names{i} '/renders/' instance_names{j}];
        mskfiles{(i-1)*length(instance_names)+j} = ['../caffe-master/data/chairs/cropped_chairs/' folder_names{i} '/masks/' instance_names{j}];
        ids((i-1)*length(instance_names)+j) = i;
        pos = strfind(instance_names{j}, '_');
        theta((i-1)*length(instance_names)+j) = str2double(instance_names{j}(pos(2)+2:pos(2)+4));
        phi((i-1)*length(instance_names)+j) = str2double(instance_names{j}(pos(3)+2:pos(3)+4));
    end
end
toc

% batching
IMG_HEIGHT = 64;
IMG_WIDTH = 64;
images = zeros(IMG_HEIGHT, IMG_WIDTH, 3, 'uint8');
masks = zeros(IMG_HEIGHT, IMG_WIDTH, 'uint8');
tic
for n = 1 : length(imfiles)
    im = imread(imfiles{n});
    if size(im,3)==1, im = cat(3,im,im,im); end
    im = imresize(im, [IMG_HEIGHT,IMG_WIDTH],'bilinear');
    im = 255-single(im);
    mask = single(rgb2gray(im)>0);
    images(:,:,:,n) = uint8(im);
    masks(:,:,n) = uint8(255*mask);
    if mod(n,62)==0,fprintf([num2str(n) ', ' num2str(toc) '\n']); end
end

% save
save('examples/chairs/data/chairs_data_64x64x3_crop.mat', 'images', 'masks', 'ids', 'phi', 'theta');

