function visualize_predictions(filename, input1, pred, input2)

input1 = permute(input1, [2,1,3]);
input1 = 1 - input1;
pred = permute(pred, [2,1,3,4]);
pred = 1 - pred;
if nargin>3,
    input2 = permute(input2, [2,1,3]);
    input2 = 1 - input2;
end

outputVideo = VideoWriter(filename);
outputVideo.FrameRate = 15;
open(outputVideo);
writeVideo(outputVideo,uint8(255*input1));
for ii = 1:size(pred,3)/3
   img = pred(:,:,(ii-1)*3+1:ii*3);
   writeVideo(outputVideo,uint8(255*img));
end
if nargin>3,
    writeVideo(outputVideo,uint8(255*input2));
end
close(outputVideo);