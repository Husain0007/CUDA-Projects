# Memory Overview

* **SGEMM** stands for Single Precision General Matrix Multiplication. 

* TroubleShooting: Resolving error associated with "helper_functions.h" header file.
    * Git cloned [cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master) and then extracted the path to the header files.
    * The header files are contained in a sub-folder called **Common**, move this to the System Include Directory.
    ```
        sudo cp -r /home/.../cuda-samples/Common/ /usr/local/include/
    ```
* **AoS** (Array of Structures) and **SoA** (Structure of Arrays). In the Structure of Arrays data is stored in separate arrays, where each array stores a single component of the data.  For example, if you were working with three-dimensional vectors, you might store the x-coordinates in one array, the y-coordinates in another array, and the z-coordinates in a third array. This is in constract to the Array of Structures (AoS), where all of the data for each element is stored together in a single contiguous block of memory. The decision t use SoA or AoS depens on the specific requirements of the application. In general, SoA tends to be more efficient for operations that only involve a subset of the data, while AoS tends to be more efficient for operations that involve the entire dataset. ([Source](https://saturncloud.io/blog/structure-of-arrays-vs-array-of-structures-in-cuda/#:~:text=This%20is%20in%20contrast%20to,for%20certain%20types%20of%20operations.))