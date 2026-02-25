# ISS - Imaging System Simulation



This project seeks to create a program capable of simulating an imaging system, which contains but is not limited to:

- An **object space** from which lights are emitted. 
- A **lens** that bends the incoming lights.
- various **attachments and modifiers** that affect how the lights travel.
- An **imager** to capture the lights and convert them into an image. 

The goal is to help animators and CG artists such that: 

- CGI sequences can be better blended with live action footage when the live action has shot with a lot of deliberately introduced image artifacts.

- Easily add believable optical effects to animations, 2D and 3D alike, while keeping the effect art-directable. 

This project does not have a clear commercial return so it would have been hard to pull off in business R&D, it would also be viewed as not professional enough in academic computer science or physics alone. A highly interdisciplinary environment would be needed for it to survive.

Special thanks to the Ohio State University, Department of Design, whose Digital Animation and Interactive Media program offered me the freedom to do this project. My gratitude also goes to the Advanced Computing Center for Art and Design, the hardware resources greatly accelerated the development of the project (imagine having a GPU with 48Gigs for graphic rams).

## Table of Content


Detailed documentation 





### Coding Conventions 

I must admit that there are a lot of inconsistencies in the coding convention here. 
For example, I started the project defaulting to my normal `UpperCaseCamel` paradigm, but later I learned that that Python is supposed to be using the `under_score_format`. 
Through the IDE I also learned that Python `if` statement does not necessarily need the parenthesis; function notes have more than one standard and `:param NAME:` is not to be taken as universal. 

As a result, there are quite a lot of variations of method/variable names and commenting styles. 

### TODO: 

- Polarzaition sometimes does not work at all 

- Add a pure brute force asph solver 


### AI 

The core lens construction and ray propagation are AI-free, not because of some moral standard, but because AI was not good enough when I wrote them. Later modules have seen much more use of AI, mostly in a "design by contract" style, as AI is asked to complete a method that fulfills certain tasks, which they perform quite well. 

