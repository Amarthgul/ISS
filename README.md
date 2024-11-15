# ISS

Imaging System Simulation. 


# 1 - General 

This project aims to establish a way to digitize lenses with physical accuracy for media production use. Digital animation can then add accurate optical characteristics in the film and live-action footages shot with vintage lenses can be matched with CGI sequences more easily.

Additionally, it may also be used to deconvolute images for information reconstruction, and in some cases, historical preservation of optics. 

## 1.1 - Background 



## 1.2 - Optical Artifact 

## 1.3 - Current Field and Similar Applications 

## 1.4 - End Goal 

The end goal of the project can be divided into several categories depending on their importance and urgency.

### 1.4.1 - "Must Accomplish"

Items in this category are mandtory for the project. The porject **must** accomplish: 

- An open-source application written in Python or C++ or C# that: 

  - Supports inputs typically seen in digital animation and visual effect productions, such as `.exr` and `.tif` images. Also supports corresponding outputs. 

  - Able to **explicitly** define an imaging system and all its parts. Include but not limited to objects, refractive/reflective surfaces, optical materials, and an imagers either ideal or based on chemical film / digital sensor. 

  - The parts are specifically designed to facilitate animation and VFX production. Meaning that it implements and emphases elements that are commonly used in these fields, such as anamorphic lens, spilt diopter. 

  - Able to propagrate an object through the virtual imaging system and accquire its image with physical accuracy. 

  - The propagation process contains both sequential and non-sequential method. 


- An algorithm that describes the mathmatical/physical process of the virtual imaging system that: 

  - For a known point source in space, could form an image of it through the virtual imaging system. 

  - The process is based on geometric optics and could perform accurate raytracing based on wavelength. 

  - The process is clear enough that people can follow this process and implement a similar program using the programming language/package they are comfortable with. 


### 1.4.2 - "Should Try to Accomplish"

Items in this category are not mandtory. **They still need to be implemented**, but due to theortical or the technical limitation, does not have to be fully operational and production ready. The porject should try to accomplish: 

- Wave optics implementations in the application. So that to simulate diffraction and its visual phenomena, such as diffraction spikes (a.k.a. sun stars). 

- A machine learning model that, when given enough info of the object and its image, could construct a "lens blackbox" that can be used in the same way as the explicit version of the applcation and simulate images of other objects. 

### 1.4.3 - Good to have

Items in this category are not mandtory. Ideally they should be included or at least attempted, but only if the previous two categories are already fulfilled. 

- A GUI interface. 

- Real-time result update when changing the parameters of the imaging system. 

## 1.5 - Examinations 



<br />

# 2 - Geometric Optics 

## 2.1 - Wavelength and RGB Conversion 

## 2.2 - Object Space 

## 2.3 - Surface

### 2.3.1 - Standard Sperical Surface 

### 2.3.2 - Even Aspherical 

### 2.3.3 - Cylindrical

## 2.4 - Object to Entrence Pupil 

## 2.5 - Sequential Propagration 

### 2.5.1 Index of Refraction 

## 2.6 - Non-Sequential Propagration 

## 2.7 - Imager 




<br />

# 3 - Waveoptics 

## 3.1 - Diffraction 


<br />

# 4 - Solving As an Inverse Problem 

