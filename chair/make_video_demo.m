addpath(genpath('../../matlab'));
h = 64;
w = 64;
files = dir('results/*_ele20_*.mat');
display = 255*ones(h*9+32,w*15,3,17,8,2,'uint8');
sample = randSample(400,135);
views = 1:4:31;
for ii = 1:9,
    for jj = 1:15,
        ind = (ii-1)*15+jj
        load(['results/' files(ind).name]);
        output = permute(output, [2,1,3,4,5]);
        output_mask = permute(output_mask,[2,1,3,4,5]);
        output = 1 - output;
        [h,w,n,m,r] = size(output);
        for vv = 1:8, % starting view
            for kk = 1:2, % direction
                for ff = 1:size(output,3)/3
                   patch = output(:,:,(ff-1)*3+1:ff*3,views(vv),kk);
		   if ff > 1,
                   mask = output_mask(:,:,ff,views(vv),kk);
		   patch = patch.*cat(3,mask,mask,mask);
		   end
                   display(32+h*(ii-1)+1:32+h*ii,w*(jj-1)+1:w*jj,:,ff,vv,kk) = uint8(255*patch);
               end
            end
        end
    end
end

save('results/display.mat','display');
