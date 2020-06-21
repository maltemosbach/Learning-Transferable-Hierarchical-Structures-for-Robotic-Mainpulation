# Learning Transferable Hierarchical Structures for Robotic Mainpulation

## New transfer learning environments

FetchPush | FetchPickAndPlace
------------ | -------------
![FetchPush_variation1-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPush_variation1-v1.png) | ![FetchickAndPlace_variation1-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPickAndPlace_variation1-v1.png)
![FetchPush_variation2-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPush_variation2-v1.png) | ![FetchickAndPlace_variation2-v1](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/FetchPickAndPlace_variation2-v1.png)

## Knowledge Visualization
![Q-value plotting tool](https://github.com/maltemosbach/Learning-Transferable-Hierarchical-Structures-for-Robotic-Mainpulation/blob/master/docs/images/Q-val_plotting1.png)
When an agent has been trained or is currently being trained its varaibales are saved in the ./models directory. By executing
```
python get_Q-plots.py
```
a script is run, which can access the current value-functions for both layers. The position of the gripper, object, subgoal and end-goal are shown in the figure. They can all be adjusted by dragging them with the mouse. Upon release, the value function for this newly created setup will be calculated and displayed. Thus, one can examine how the agent values states for different situations and analyze its knowledge representation. This is most useful for the pushing tasks, since they can be viewed as mainly 2-dimensional problems. The specifics of the variations are also hinted at in the visualization of the tabletop. 

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
-np n | Run with n processes
--test | Only testing
--show | Display the agent
--transfer | Transfer a pretrained lower layer
--contin | Continue training with the learned agent

By executing the default
```
python run.py
```
command, the agent will alternate between training and testing. if multiple runs or hyperparameter-combinations are given they will be run successively. To view a trained agent, its saved parameters must be in the ./models directory in HAC. The command
```
python run.py --test --show
```
will then display the trained agent.
