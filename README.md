# ISS - Imaging System Simulation

This repo is the proof of concept implementation of my thesis, which seeks to create a framework capable of simulating an imaging system consists of at least:

- An **object space** from which lights are emitted. 
- A **lens** that bends the incoming lights.
- various **attachments and modifiers** that affect how the lights travel.
- An **imager** to capture the lights and convert them into an image. 

The goal is to help animators and CG artists such that: 

- CGI sequences can be better blended with live action footage when the live action has shot with a lot of deliberately introduced image artifacts.

- accurate optical effects can be easily added into animations, 2D and 3D alike, while keeping the effect art-directable. 

Additionally, it could also:

- help computer vision and AI researchers to generate "ground truth" image results without physically possessing the optical instruments. 

- help lens and optics enthusiasts to gain more insights of photographic objects by offering them a way to perform 100% repeatable and reproducible experiments. 

This project does not have a clear commercial return model so it would have been hard to pull off in business R&D, it would also be viewed as not cutting-edge enough in academic computer science or physics alone. A highly interdisciplinary environment would be needed for it to survive. Special thanks thus goes to the Ohio State University, Department of Design, whose Digital Animation and Interactive Media program somehow allowed me to do whatever I want. My gratitude also goes to the Advanced Computing Center for Art and Design, the hardware resources greatly accelerated the development of the project _(imagine casually playing with a GPU with 48Gigs for graphic rams)_.


## Important notes on using the project 

This repo is, again, a proof of concept for my thesis, it had two prior versions (one of which was in C++) and is built gradually in a 3-year period. As such, there are a lot of places that could have be designed and coded in better ways but had to stay as they are due to connectivity and budget reasons. 

As such, **I do not recommend directly using this repo for production**. If you are a production studio, reference the framework documentation, use your technical team and AI to rewrite it _(ideally not in Python) (the documentation might still be WIP, but it's totally possible to feed this repo to AI and let them parse it)_. 


## Coding Conventions 

I must admit that there are a lot of inconsistencies in the coding convention here. 
For example, likely due to my prior habit with `C++` and `C#`, I started the project defaulting to my normal `UpperCaseCamel` naming paradigm, but later I learned that that Python is supposed to be using the `under_score_format`... 
Through the IDE (thanks, PyCharm) I also learned that Python `if` statement does not necessarily need the parenthesis; and that function notes have more than one standard and `:param NAME:` is not to be taken as universal. 

As a result, there are quite a lot of variations of method/variable names and commenting styles. 

## TODO: 

- Add a pure brute force ASPH solver. 

- Add channel summing in the imager. 

- Add opacity based on image staking. 

- Add propagation based film grain. 

- Add onion rings in ASPH bokeh.

- Add transform for gamma corrected inputs. 

- Finish conical elements. 

## AI 

The core lens construction and ray propagation are AI-free, not because of some moral standard, but because AI was not good enough when I wrote them. Later modules have seen much more use of AI, mostly in a "design by contract" style, as AI is asked to complete a method that fulfills certain tasks, which they perform quite well. 


## Future work

This repo is still largely based on geometric optics, with the primary improvement being a comprehensive spectral tracing. 

The consequence of not embracing full physical optics is that effects like diffraction and interference do not naturally occur. Consequently, image phenomena like sun star and bokeh dust cannot be easily recreated. 

The next step apparently is to devise a way to perform full image simulation based on physical optics. Technically this is already possible, but the computational cost is several orders of magnitude higher than this framework, the bulk of research work would then be about how to represent the wavefront so that it is cheap to calculate while still highly accurate. 

However, purely computer graphics wise, traditional 3D representation is reseeding, with radiant field, splats, and sometimes direct image-to-image approach taking over certain applications. The next step for computer graphics at large is likely a scene representation that directly engages with the physical characteristic of objects, rather than the mere surface radiometry behavior. This new model could be designed for more than just visual experience, but other forms of interaction as well, such as touch or sound. 

Of course, there would then be the question of whether replication is needed at all. Daß alle unsere Erkenntnis mit der Erfahrung anfange, if the goal is to invoke a certain feeling or acquiring certain recognition, it would be much more effective to directly manipulate the subjective perception via biological intervention than to painstakingly recreate an accurate representation of the objective world. 

