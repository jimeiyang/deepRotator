% prepare batches for chairs

load('data/chairs/rendered_chairs/all_chair_names.mat');
for i=1:length(folder_names)
    tic
    images = cell(length(instance_names),1);
    imfiles = cell(length(instance_names),1);
    bboxes = zeros(length(instance_names),4);
    for j=1:length(instance_names)
        imfiles{j} = ['data/chairs/rendered_chairs/' folder_names{i} '/renders/' instance_names{j}];
        im = imread(imfiles{j});
        images{j} = im;
	mask = rgb2gray(im)>0;
        xx = find(sum(mask,1)); xmin = xx(1); xmax = xx(end);
        yy = find(sum(mask,2)); ymin = yy(1); ymax = yy(end);
        bboxes(j,:) = [xmin,xmax,ymin,ymax];
    end
    x1 = min(bboxes(:,1)); x2 = max(bboxes(:,2)); 
    y1 = min(bboxes(:,3)); y2 = max(bboxes(:,4));
    cx = floor((x1+x2)/2); cy = floor((y1+y2)/2);
    h = y2-y1+1; w = x2-x1+1;
    h1 = floor(h/2); h2 = h-h1;
    w1 = floor(w/2); w2 = w-w1;
    if h > w,
        x1 = cx-h1; x2 = cx+h2-1;
    elseif h < w,
        y1 = cy-w1; y2 = cy+w2-1;
    end
    %
    mkdir(['data/chairs/cropped_chairs/' folder_names{i} '/renders']);
    batch = zeros(128,128,3,62,'uint8');
    for j=1:length(instance_names)
        im = images{j};
        [im_h,im_w,~] = size(im);
        pad = [1-x1,x2-im_w,1-y1,y2-im_h];
        im = imPad(im,pad,255);
        im = imPad(im,20,255);
        im = imresize(im,[128,128],'bilinear');
        batch(:,:,:,j) = im;
        imfile = strrep(imfiles{j},'rendered_chairs','cropped_chairs');
        imwrite(im,imfile,'png');
    end
%     prm = struct('hasChn',1);
%     montage2(batch,prm); drawnow;
    disp(sprintf('done with class %d in %f seconds',i,toc));
end


