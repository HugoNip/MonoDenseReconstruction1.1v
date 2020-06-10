## Introduction
This project shows how to compute a corresponding point 
in other images base on the point in reference image. 
We have known the accurate pose of camera (T).

### Reference Point
![reference_circle.png](https://github.com/HugoNip/MonoDenseReconstruction2.0v/blob/master/results/reference_circle.png)

### Estimated Point
![final_circle_0001.png](https://github.com/HugoNip/MonoDenseReconstruction2.0v/blob/master/results/final_circle_0001.png)

## Requirements

### Eigen Package (Version >= 3.0.0)
#### Source
http://eigen.tuxfamily.org/index.php?title=Main_Page

#### Compile and Install
```
cd [path-to-Eigen]
mkdir build
cd build
cmake ..
make 
sudo make install 
```
#### Search Installing Location
```
sudo updatedb
locate eigen3
```

default location "/usr/include/eigen3"

### OpenCV
#### Required Packages
OpenCV  
OpenCV Contrib

gcc version: gcc (Ubuntu 5.4.0-6ubuntu1/~16.04.12) 5.4.0 20160609   
g++ version: g++ (Ubuntu 5.4.0-6ubuntu1/~16.04.12) 5.4.0 20160609  
OpenCV with version of both 3.0 and 4.0 tests well.   
(Note: OpenCV will fail to compile with gcc/g++ of 9.2.0 version)

### Sophus Package
#### Download
https://github.com/HugoNip/Sophus

#### Compile and Install
```
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
```


## Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

## Run
```
./build/dense_mono
```


## Reference
[Source](https://github.com/HugoNip/MonoDenseReconstruction1.0v)
