# deepRotator
This is the code for NIPS15 paper [Weakly-supervised disentangling with recurrent transformations for 3D view synthesis](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf) by Jimei Yang, Scott Reed, Ming-Hsuan Yang and Honglak Lee.

Please follow the instructions below to run the code.

1. Download the preprocessed chair data from "https://www.dropbox.com/s/q2bih317hyhbpm5/chairs_data_64x64x3_crop.mat?dl=0", and save it to the "./chair/data/" folder.
2. Compile the caffe and matcaffe in "./caffe-cedn/" (Makefile.config needs to be adjusted according to your machine).
3. Train the model by running the matlab scripts "train_chair_rotator_base.m" and then "train_chair_rotator_curriculum.m".
4. The pre-trained RNN16 model for chair rotation can be downloaded from "https://www.dropbox.com/s/h7iiei53u2g1vvn/rnn_t16_model.tar.gz?dl=0".

The single-view chair rotation demo can be found in the Youtube: 

<a href="https://www.youtube.com/watch?v=3dPwiWnDoNY" target="_blank"><img src="https://github.com/jimeiyang/deepRotator/blob/master/demo_img.png" 
alt="IMAGE ALT TEXT HERE" width="640" height="360" border="10" /></a>

Please contact "jimyang@adobe.com" if any questions. 
