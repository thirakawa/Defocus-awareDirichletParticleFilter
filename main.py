# -*- coding: utf-8 -*-
"""
Copyright (C) 2016 by Tsubasa Hirakawa
hirakawa@eml.hiroshima-u.ac.jp

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import dirichletParticleFilter as dpf

if __name__ == '__main__':

    # import observation file and IPR file as numpy array
    observations = np.loadtxt("obs.csv", delimiter=',')
    confidence = np.loadtxt("confidence.csv", delimiter=',')

    # compare data length and dimension
    dimension = observations.shape[1]
    data_length = observations.shape[0]
    IPR_length = confidence.shape[0]
    if data_length != IPR_length:
        print "error: data length is different !"
        sys.exit(-1)

    # initialize DPF by initial observation
    pf = dpf.DPF(theta=100, gamma=1, nPar=100)
    ddpf = dpf.DDPF(theta=100, gamma=1, nPar=100)
    pf.initialize(observations[0, :])
    ddpf.initialize(observations[0, :])

    # print parameters of DPF and DDPF
    pf.print_parameters()
    ddpf.print_parameters()

    # smoothing
    dpf_res  = observations.copy()
    ddpf_res = observations.copy()

    for i in range(data_length):
        # update
        pf.update(observations[i, :])
        ddpf.update(observations[i, :], confidence[i])
        # parameter estimation
        pf.estimateParameter()
        ddpf.estimateParameter()
        # compute mode
        dpf_res[i, :]  = pf.mode()
        ddpf_res[i, :] = ddpf.mode()

    # plot
    plt.subplot(4,1,1)
    plt.plot(observations)
    plt.ylim(0, 1)
    plt.title("observation")
    plt.subplot(4,1,2)
    plt.plot(dpf_res)
    plt.ylim(0, 1)
    plt.title("dirichlet particle filter")
    plt.subplot(4,1,3)
    plt.plot(ddpf_res)
    plt.ylim(0, 1)
    plt.title("defocus-aware dirichlet particle filter")
    plt.subplot(4,1,4)
    plt.plot(confidence)
    plt.title("confidence values")
    plt.tight_layout()
    plt.show()
