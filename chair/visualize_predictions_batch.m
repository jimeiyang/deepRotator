function visualize_predictions_batch(filename, input)

input = permute(input, [2,1,3,4,5]);
input = 1 - input;
[h,w,n,m,r] = size(input);

outputVideo = VideoWriter(filename);
outputVideo.FrameRate = 5;
open(outputVideo);
for kk = 1:2,
  strs = cell(1,17); strs{1} = 'Input';
  for ii=1:16, strs{ii+1} = sprintf('Clockwise rotation, t = %d', ii); end
for ff = 1:size(input,3)/3
   display = ones(h*4+32,w*6,3);
   insertText(display, [w*2,5], strs{ff} );
   for ii = 1:4,
       for jj = 1:6,
           patch = input(:,:,(ff-1)*3+1:ff*3,(ii-1)*4+jj,kk);
           display(32+h*(ii-1)+1:32+h*ii,w*(jj-1)+1:w*jj,:) = patch;
       end
   end
  writeVideo(outputVideo,uint8(255*display));
end
end
close(outputVideo);
