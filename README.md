# Learning Transferable Hierarchical Structures for Robotic Mainpulation

## New transfer learning environments

FetchPush | FetchPickAndPlace
------------ | -------------
![FetchPush_variation1-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPush_variation1-v1.png) | ![FetchickAndPlace_variation1-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPickAndPlace_variation1-v1.png)
![FetchPush_variation2-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPush_variation2-v1.png) | ![FetchickAndPlace_variation2-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPickAndPlace_variation2-v1.png)

## Knowledge visualiozation
(Add image and explanation here)

## Setup
The modified Gym suite containing the additional transfer learning environments depicted above is included in the setup directory of this Repo as Gym.zip. After downloading the file it must be installed by moving to the gym directory and installing it via pip:

```
pip install -e .
```

Further, the used Conda environment is included in the setup directory. 

## Running experiments
The starting file for all experiments is the run.py inside HAC. Inside the file miscellaneous hyperparameters can be defined. Secondly, flags can be used to alter how the experiments are run.

FLAG | Explanation
------------ | -------------
-np 4 | Run with 4 processes
--test | Only testing
--show | Display the agent
--transfer | Transfer a pretrained lower layer

By executing the default
```
python run.py
```
command, the agent will alternate between training and testing. if multiple runs or hyperparameter-combinations are given they will be run successively. To view a trained agent, its saved parameters must be in the ./models directory in HAC. The command
```
python run.py --test --show
```
will then display the trained agent.
