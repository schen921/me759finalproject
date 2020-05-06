# me759finalproject
ME 759 Final Project

Compile convolution neural network:

nvcc network.cu cnn.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o cnn

Run:

./cnn

The script will automatically run 20 epoches in a batch size of 100 among 10000 data samples. And show the cross entropy,
count of how many of prediction label and actual label match of the last batch.




Compile convolution/acclerated convolution:

nvcc network.cu convolution.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o convolution

Run:

./convolution number mode

The first argument "number" is the number of images.
The second argument "mode" is the mode of convolution. When mode = 0, it is normal convolution, otherwise it is acclerated convolution
The script will return the time it consumed.
