# -*- coding: utf-8 -*-
"""
Copyright (C) 2016 by Tsubasa Hirakawa
hirakawa@eml.hiroshima-u.ac.jp

"""

import math
import numpy as np
import scipy as sp
import dirichlet as diri

class DPF(object):
    """
    Dirichlet Particle Filter
    """

    def __init__(self, theta=100.0, gamma=1.0, theta_bias=0.0, gamma_bias=1.0, nPar=100):
        """
        Parameters
        ----------
        theta: scale parameter for state transition
        theta_bias: bias term for state transition
        gamma: scale parameter for likelihood
        gamma_bias: bias term for likelihood
        nPar: number of particles
        """
        super(DPF, self).__init__()
        self.theta = theta
        self.gamma = gamma
        self.theta_bias = theta_bias
        self.gamma_bias = gamma_bias
        self.numParticle = nPar

    def initialize(self, initialDist):
        self.dimension = int( initialDist.shape[0] )
        alpha = self._calc_alpha(initialDist, self.gamma, self.gamma_bias)
        self.particles = np.random.dirichlet(alpha, self.numParticle)

    def update(self, observation):
        if observation.shape[0] != self.dimension:
            print "warning: dimension of observation array is differ."
            print "         dimension is", self.dimension
        # transition
        for i in xrange(0, self.numParticle):
            alpha = self._calc_alpha(self.particles[i, :], self.theta, self.theta_bias)
            self.particles[i, :] = np.random.dirichlet(alpha, 1)[0]
        # weighting
        alpha = self._calc_alpha(observation, self.gamma, self.gamma_bias)
        weights = np.zeros(self.numParticle, dtype='float')
        for i in xrange(0, self.numParticle):
            weights[i] = sp.stats.dirichlet.pdf(self.particles[i, :], alpha)
        # normalize weights
        weights = weights / weights.sum()
        # resampling
        sampledIndex = np.random.choice(range(self.numParticle), self.numParticle, p=weights)
        nextParticles = self.particles.copy()
        for i, index in enumerate(sampledIndex):
            nextParticles[i, :] = self.particles[index, :]
        self.particles = nextParticles.copy()

    def _calc_alpha(self, inputArr, scale, bias):
        return inputArr * scale + bias

    def estimateParameter(self):
        self.alpha = diri.mle(self.particles)
        return self.alpha

    def mean(self):
        return self.alpha / self.alpha.sum()

    def variance(self):
        a0 = self.alpha.sum()
        return self.alpha * ( a0 - self.alpha ) / ( a0*a0*(a0+1) )

    def mode(self):
        if np.sum(self.alpha > 1) != self.dimension:
            print "warning: some of the element of alpha may be less than 1."
            print "         cannot compute mode."
            return np.zeros(self.dimension, dtype='float')
        a0 = self.alpha.sum()
        return ( self.alpha - 1.0 ) / ( a0 - float(self.dimension) )

    def print_parameters(self):
        print "Dirichlet Particle Filter ===================="
        print "Parameters"
        print "    number of particle:", self.numParticle
        print "    dimension:", self.dimension
        print "    theta:", self.theta
        print "    gamma:", self.gamma
        print "    state transition bias:", self.theta_bias
        print "    likelihood bias:", self.gamma_bias
        print "=============================================="

    def set_theta(self, theta):
        self.theta = float(theta)
        
    def set_gamma(self, gamma):
        self.gamma = float(gamma)
        
    def set_particle_number(self, nPar):
        self.numParticle = nPar


class DDPF(DPF):
    """
    Defocus-aware Dirichlet Particle Filter    
    """
    
    def update(self, observation, confidence):
        if observation.shape[0] != self.dimension:
            print "warning: dimension of observation array is differ."
            print "         dimension is", self.dimension

        # transition
        for i in xrange(0, self.numParticle):
            alpha = self._calc_alpha(self.particles[i, :], self.theta, self.theta_bias)
            self.particles[i, :] = np.random.dirichlet(alpha, 1)[0]

        # weighting
        scaledConfidence = self._scaleConfidence(confidence)
        alpha = self._calc_alpha(observation, np.random.rayleigh(scaledConfidence, 1)[0], self.gamma_bias)
        weights = np.zeros(self.numParticle, dtype='float')
        for i in xrange(0, self.numParticle):
            weights[i] = sp.stats.dirichlet.pdf(self.particles[i, :], alpha)
        # normalize weights
        weights = weights / weights.sum()

        # resampling
        sampledIndex = np.random.choice(range(self.numParticle), self.numParticle, p=weights)
        nextParticles = self.particles.copy()
        for i, index in enumerate(sampledIndex):
            nextParticles[i, :] = self.particles[index, :]
        self.particles = nextParticles.copy()

    def _scaleConfidence(self, confidence):
        return 4 * math.exp(100 * math.log(0.25) * confidence)
    
    def print_parameters(self):
        print "defocus-aware Dirichlet Particle Filter ======"
        print "Parameters"
        print "    number of particle:", self.numParticle
        print "    dimension:", self.dimension
        print "    theta:", self.theta
        print "    gamma (for init):", self.gamma
        print "    state transition bias:", self.theta_bias
        print "    likelihood bias:", self.gamma_bias
        print "=============================================="
