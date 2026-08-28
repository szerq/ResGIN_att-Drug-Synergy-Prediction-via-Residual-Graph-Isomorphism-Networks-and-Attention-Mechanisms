# ResGIN-att
We introduce a novel model named the Residual Graph Isomorphism Network integrated with an Attention mechanism (ResGIN-Att). The model first extracts multi scale topological features of drug molecules using a residual graph isomorphism network, where residual connections help mitigate over-smoothing in deep layers. Subsequently, an adaptive LSTM module fuses structural information from local to global scales. Finally, a cross-attention module is designed to explicitly model drug-drug interactions and identify key chemical substructures.Once the paper is accepted, we will immediately make the code available.
![image](https://github.com/szerq/ResGIN-att/blob/main/model.png)
# Data
Download data file from [here](https://drive.google.com/file/d/1Gqt4HxbUVILIbp17L6e_zLGA_3sVKOw1/view?usp=sharing), and put it into data directory  
# Requirements
Python 3.9.18                                                                                                                                                                                     
Pytorch 2.2.0                                                                                                                                                                                     
Pytorch Geometric 2.6.1                                                                                                                                                                           
numpy 1.26.4
# Run
main.py
# Result
![image](https://github.com/szerq/ResGIN-att/blob/main/result.png)
