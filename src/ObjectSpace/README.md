


# Class inheritance order tree 

- **Image2D**

  This is the general interface, very little feature is included. 

  - **Image2DFlat**
  
    Basic feature of loading a 2D image and place it in a spatial position. 

  - **Image2DVariDepth**
  
    Can read EXR and its channel info.

    - **Image2DVariHighlightExtension**
    
      Can take a range of depth. 

      - **Image2DFlatHighlightExtension**

        Takes a flat depth value. 
