# Defocus-aware Dirichlet Particle Filter (D-DPF)

This program includes Dirichlet Particle filter[1, 2] and Defocus-aware Dirichlet Particle Filter[2].

For parameter estimation of Dirichlet distribution, [Dirichlet MLE python library](https://github.com/ericsuh/dirichlet) is used.
Greatly thanks to his work.
Note that this parameter estimation is different to ones written in the paper [2].


## Execution environment
* OS: Mac OS X (10.11)
* Language: Python 2.7 (Anaconda 2.4.*)
* Modules: Numpy, Scipy, Matplotlib


## Usage
    python ./main.py


## References
1. T. Hirakawa, T. Tamaki, B. Raytchev, K. Kaneda, T. Koide, Y. Kominami, R. Miyaki, T. matsuo, S. Yoshida, S. Tanaka, "Smoothing posterior probabilities with a particle filter of Dirichlet distribution for stabilizing colorectal NBI endoscopy recognition," In Proc. of IEEE International Conference on Image Processing (ICIP2013), pp. 621-625, 2013.
2. T. Hirakawa, T. Tamaki, B. Raytchev, K. Kaneda, T. Koide, S. Yoshida, Y. Kominami, S. Tanaka, "Defocus-aware Dirichlet particle filter for stable endoscopic video frame recognition," Artificial Intelligence in Medicine, 2016. DOI:[http://dx.doi.org/10.1016/j.artmed.2016.03.002](http://dx.doi.org/10.1016/j.artmed.2016.03.002).
