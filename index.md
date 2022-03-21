# PINO Applications

In this work, we examine the applications of physics informed neural operators (PINOs).  PINOs have demonstrated excellent ability to reproduce results of various test simulations.  Here we stress test PINOs over a wide range of problems including the variations of the wave equation, Burgers equation and the shallow water equations.  The source code for this work can be found at [this repo](https://github.com/shawnrosofsky/PINO_Applications). 
We also provide users with a way to try out our code at Argonne's [Data and Learning Hub for Science](https://www.dlhub.org).

## Abstract

## Methods

## Results

### Wave Equation 1D
The 1D wave equation was the first test for our PINOs.  This equation computationally simple PDE that is second order in time and models a variety of different physics phenomena.  This equation is given by

![Equation: Wave Equation 1D](http://www.sciweavers.org/download/Tex2Img_1647640969.jpg)

where c=1 is our wave speed and with periodic boundary conditions.  

We present results bellow illustrating the ability of the PINO to reconstruct the simulated result for multiple initial conditions.  The differences between the simulated data and the PINO are visually indistinguishable.

![Wave Equation 1D 0](assets/movies/Wave1D_0.gif) ![Wave Equation 1D 1](assets/movies/Wave1D_1.gif) ![Wave Equation 1D 2](assets/movies/Wave1D_2.gif)



### Wave Equation 2D
![Wave Equation 2D 0](assets/movies/Wave2D_0.gif)
<!-- ![Wave Equation 2D 1](assets/movies/Wave2D_1.gif) -->

### Burgers Equation 1D
![Burgers Equation 1D 0](assets/movies/Burgers1D_0.gif) ![Burgers Equation 1D 1](assets/movies/Burgers1D_1.gif) ![Burgers Equation 1D 2](assets/movies/Burgers1D_2.gif)

### Burgers Equation 2D Scalar
![Burgers Equation 2D 2](assets/movies/Burgers2D_2.gif)
<!-- ![Burgers Equation 2D 3](assets/movies/Burgers2D_3.gif) -->

### Burgers Equation 2D Inviscid
![Burgers Equation 2D Inviscid 2](assets/movies/Burgers2D_novisc_2.gif)
<!-- ![Burgers Equation 2D Inviscid 3](assets/movies/Burgers2D_novisc_3.gif) -->

### Burgers Equation 2D Vector
![Burgers Equation 2D Vector u](assets/movies/Burgers2D_coupled_u.gif)
<!-- ![Burgers Equation 2D Vector v](assets/movies/Burgers2D_coupled_v.gif) -->

<!-- ### Linear Shallow Water Equations 2D
![Linear Shallow Water Equations 2D h](assets/movies/SWE_Linear_f1_h.gif)
![Linear Shallow Water Equations 2D u](assets/movies/SWE_Linear_f1_u.gif)
![Linear Shallow Water Equations 2D v](assets/movies/SWE_Linear_f1_v.gif) -->

### Nonlinear Shallow Water Equations 2D
![Nonlinear Shallow Water Equations 2D eta](assets/movies/SWE_Nonlinear_eta.gif)
![Nonlinear Shallow Water Equations 2D u](assets/movies/SWE_Nonlinear_u.gif)
![Nonlinear Shallow Water Equations 2D v](assets/movies/SWE_Nonlinear_v.gif)



<!-- Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
 -->
