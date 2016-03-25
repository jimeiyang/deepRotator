# deepRotator
This is the code for NIPS15 paper [Weakly-supervised disentangling with recurrent transformations for 3D view synthesis](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf) by Jimei Yang, Scott Reed, Ming-Hsuan Yang and Honglak Lee.

Please follow the instructions below to run the code.
1. Download the preprocessed chair data from "https://dl.dropboxusercontent.com/u/2885859/chairs_data_64x64x3_crop.mat", and save it to the "./chair/data/" folder.
2. Compile the caffe and matcaffe in "./caffe-cedn/" (Makefile.config needs to be adjusted according to your machine).
3. Train the model by running the matlab scripts "train_chair_rotator_base.m" and then "train_chair_rotator_curriculum.m".

The single-view chair rotation demo can be found in the Youtube: <a href="https://www.youtube.com/watch?v=3dPwiWnDoNY" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

Please contact "jimyang@adobe.com" if any questions. 
