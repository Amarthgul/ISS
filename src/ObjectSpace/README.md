


# Class inheritance order tree 

- **Image2D**

  This is the general interface, very little feature is included. 

  - **Image2DFlat**
  
    Basic feature of loading a 2D image and place it in a spatial position as a flat image. 

  - **Image2DVariDepth**
  
    Can read EXR and its channel info, including alpha, depth, and other AOVs. 

    - **Image2DVariHighlightExtension**
    
      Can take a range of depth by setting the `zDepthMappingRange`. It inherits from `Image2DVariDepth` exactly to solve this depth-ray intersection check. 

      - **Image2DFlatHighlightExtension**

        Takes a flat depth value. It overrides the depth-ray intersection check from `Image2DVariHighlightExtension` and replaced it with a much simpler close form ray-plane intersection solution. 
