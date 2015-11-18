This software is freely available for academic use only. 
You should not distribute this software without having first my permission. 

The code is based on the paper:

M. K. Titsias and M. Lazaro-Gredilla.
Spike and Slab Variational Inference for Multi-Task 
and Multiple Kernel Learning, NIPS, 2011. 

The main functions for mult-task multiple kernel learning
are: varmkgpCreate.m,  varmkgpTrain.m, varmkgpPredict.m 
while varmkgpMissDataTrain.m deals with training with missing
values. See the demos.

There exist also an implementation of variational spike and slab 
multiple output linear regression realized by the functions:
slrCreate.m and slrPairMF.m; see demo_linreg.m. 

All the above main software was written by myself  
(with some modification done by Miguel).

The misc_toolbox contains auxiliary functions 
most of them written by Miguel. It also  
contains the 'gpml' toolbox of Rasmussen and Williams
which is used by our code. 

Michalis K. Titsias, October, 2011

