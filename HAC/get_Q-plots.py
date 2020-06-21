"""
Add text ...
"""
from utils import _get_combinations
from options import parse_options
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib as mpl
from environment import Environment
from agent import Agent
from tensorboardX import SummaryWriter
import tensorflow as tf
import numpy as np
import matplotlib.patches as patches

# TODO: Switching between interpolation and raw data

clicked_on = "nothing"

def button_click(event):
    global number_of_layers_displayed, fig, axs, switch_disp_layers
    if number_of_layers_displayed == 2:
        axs[0].set_visible(False)
        number_of_layers_displayed = 1
        axs[1].set_position([0.1, 0.404, 0.8, 0.476])
        switch_disp_layers.label.set_text("Show both layers")
    elif number_of_layers_displayed == 1:
        axs[0].set_visible(True)
        axs[1].set_position([0.5363636363636364, 0.404, 0.3636363636363636363, 0.476])
        number_of_layers_displayed = 2
        switch_disp_layers.label.set_text("Show only higher layer")

def on_press(event):
    global clicked_on
    xpress, ypress = event.xdata, event.ydata

    if xpress != None and ypress != None:
        if xpress >= object_pos_proj[0]-1 and xpress <= object_pos_proj[0]+1 and ypress >= object_pos_proj[1]-1 and ypress <= object_pos_proj[1]+1:
            clicked_on = "object"
        elif xpress >= gripper_pos_proj[0]-1 and xpress <= gripper_pos_proj[0]+1 and ypress >= gripper_pos_proj[1]-1 and ypress <= gripper_pos_proj[1]+1:
            clicked_on = "gripper"
        elif xpress >= goal_pos_proj[0]-1 and xpress <= goal_pos_proj[0]+1 and ypress >= goal_pos_proj[1]-1 and ypress <= goal_pos_proj[1]+1:
            clicked_on = "goal"
        elif xpress >= subgoal_pos_proj[0]-1 and xpress <= subgoal_pos_proj[0]+1 and ypress >= subgoal_pos_proj[1]-1 and ypress <= subgoal_pos_proj[1]+1:
            clicked_on = "subgoal"
        else:
            clicked_on = "nothing"
    else:
        clicked_on = "nothing"


    fig.canvas.draw()


def on_release(event):
    global clicked_on
    global object_pos_proj
    global gripper_pos_proj
    global goal_pos_proj
    global subgoal_pos_proj
    global object_pos
    global gripper_pos
    global goal_pos
    global subgoal_pos
    global Q_vals_layer_1
    xpress, ypress = event.xdata, event.ydata

    if clicked_on == "nothing":
        pass
    else:
        if clicked_on == "object":
            object_pos_proj[0] = xpress
            object_pos_proj[1] = ypress
            for i in range(2):
                object[i].set_xy((object_pos_proj[0]-1, object_pos_proj[1]-1))
            object_pos = proj_mpl2env(object_pos_proj)

        elif clicked_on == "gripper":
            gripper_pos_proj[0] = xpress
            gripper_pos_proj[1] = ypress
            for i in range(2):
                gripper_1[i].set_xy((gripper_pos_proj[0]-1, gripper_pos_proj[1]-1))
                gripper_2[i].set_xy((gripper_pos_proj[0], gripper_pos_proj[1]-1))
            gripper_pos = proj_mpl2env(gripper_pos_proj)

        elif clicked_on == "goal":
            goal_pos_proj[0] = xpress
            goal_pos_proj[1] = ypress
            for i in range(2):
                goal[i].set_xy((goal_pos_proj[0] - 1, goal_pos_proj[1] - 1))
            goal_pos = proj_mpl2env(goal_pos_proj)

        elif clicked_on == "subgoal":
            subgoal_pos_proj[0] = xpress
            subgoal_pos_proj[1] = ypress
            subgoal.set_xy((subgoal_pos_proj[0] - 1, subgoal_pos_proj[1] - 1))
            subgoal_pos = proj_mpl2env(subgoal_pos_proj)

        Q_vals_layer_0, Q_vals_layer_1 = generate_Q_map(object_pos, gripper_pos, goal_pos, subgoal_pos)

        im[0].set_data(Q_vals_layer_0)
        im[1].set_data(Q_vals_layer_1)

    fig.canvas.draw()

def generate_Q_map(object_pos, gripper_pos, goal_pos, subgoal_pos):
    if env.name == "FetchPush-v1" or env.name == "FetchPush_variation1-v1" or env.name == "FetchPush_variation2-v1" or env.name == "FetchPickAndPlace-v1" or env.name == "FetchPickAndPlace_variation1-v1" or env.name == "FetchPickAndPlace_variation2-v1":
        Q_vals_layer_0 = np.ones((20, 28))
        Q_vals_layer_1 = np.ones((20, 28))

        # Layer 0
        g = subgoal_pos
        o = np.empty((20, 28, 25))
        u = np.zeros(4)
        for i in range(20):
            for j in range(28):
                # Vary object position over the map
                o[i, j, :] = np.concatenate((gripper_pos, np.array([1.0625 + i*0.025, 0.4125 + j*0.025, 0.425]), np.zeros(19)))
                if agent.layers[0].policy is not None:
                    Q_vals_layer_0[i, j] = agent.layers[0].policy.get_Q_values_pi(o[i, j, :], g, u,
                                                                                 use_target_net=False)

        del g, o, u

        # Layer 1
        g = goal_pos
        o = np.concatenate((gripper_pos, object_pos, np.zeros(19)))
        u = np.empty((20, 28, 3))
        for i in range(20):
            for j in range(28):
                u[i, j, :] = np.array([1.0625 + i * 0.025, 0.4125 + j * 0.025, 0.425])
                if agent.layers[1].policy is not None:
                    Q_vals_layer_1[i, j] = agent.layers[1].policy.get_Q_values_u(o, g, u[i, j, :],
                                                                                 use_target_net=False)
                elif agent.layers[1].critic is not None:
                    Q_vals_layer_1[i, j] = agent.layers[1].critic.get_target_Q_value(np.reshape(o, (1, 25)),
                                                                                     np.reshape(g, (1, 3)),
                                                                                     np.reshape(u[i, j, :], (1, 3)))

    else:
        assert False

    return Q_vals_layer_0, Q_vals_layer_1

def proj_env2mpl(pos):
    assert pos.shape[0] == 3
    proj_pos = np.zeros(3)
    proj_pos[0] = (((pos[1] - 0.4) / 0.7) *28) - 0.5
    proj_pos[1] = (((pos[0] - 1.05) / 0.5) *20) - 0.5
    proj_pos[2] = pos[2]
    return proj_pos

def proj_mpl2env(pos):
    assert pos.shape[0] == 3
    proj_pos = np.zeros(3)
    proj_pos[0] = (((pos[1] + 0.5) / 20) *0.5) + 1.05
    proj_pos[1] = (((pos[0] + 0.5) / 28) *0.7) + 0.4
    proj_pos[2] = pos[2]
    return proj_pos



FLAGS = parse_options()

FLAGS.contin = True


hyperparameters = {
    "env": ['FetchPush_variation2-v1'],
    "ac_n": [0.2],
    "sg_n": [0.2],
    "replay_k": [4],
    "layers": [2],
    "use_target": [[False, True]],
    "sg_test_perc": [0.1],
    "buffer": [['transitions', 'transitions']],
    "samp_str": ['HAC'],
    "modules": [['baselineDDPG', 'actorcritic']]

}

FLAGS.time_scale = 10
FLAGS.max_actions = 50

FLAGS.subgoal_penalty = -FLAGS.time_scale
save_models = False

hparams = _get_combinations(hyperparameters)
hparams = hparams[0]

sess = tf.compat.v1.InteractiveSession()
writer_graph = tf.compat.v1.summary.FileWriter("./temp_Q")
writer = SummaryWriter("./temp_Q")

# Create agent and environment
if hparams["env"] == "FetchReach-v1":
    project_state_to_end_goal = lambda sim, state: state[0:3]
    project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[0] > 1.55 else 1.05 if state[0] < 1.05 else state[0], 1.1 if state[1] > 1.1 else 0.4 if state[1] < 0.4 else state[1], 1.1 if state[2] > 1.1 else 0.4 if state[2] < 0.4 else state[2]])
elif hparams["env"] == "FetchPush-v1" or hparams["env"] == "FetchPush_variation1-v1" or hparams["env"] == "FetchPush_variation2-v1" or hparams["env"] == "FetchPickAndPlace-v1" or hparams["env"] == "FetchPickAndPlace_variation1-v1" or hparams["env"] == "FetchPickAndPlace_variation2-v1":
    project_state_to_end_goal = lambda sim, state: state[3:6]
    project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[3] > 1.55 else 1.05 if state[3] < 1.05 else state[3], 1.1 if state[4] > 1.1 else 0.4 if state[4] < 0.4 else state[4], 1.1 if state[5] > 1.1 else 0.4 if state[5] < 0.4 else state[5]])
else:
    assert False, "Unknown environment given."

dist_threshold = 0.05
end_goal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])
subgoal_bounds = np.array([[1.05, 1.55], [0.4, 1.1], [0.4, 1.1]])
subgoal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])

# Instantiate and return agent and environment
env = Environment(hparams["env"], project_state_to_end_goal, end_goal_thresholds, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, FLAGS.max_actions, FLAGS.show)
agent = Agent(FLAGS,env, writer, writer_graph, sess, hparams)

# Define position of object, gripper and goal for the critic to evaluate
object_pos = np.array([1.15, 0.55, 0.425])
gripper_pos = np.array([1.25, 0.45, 0.425])
goal_pos = np.array([1.25, 0.88, 0.425])
subgoal_pos = np.array([1.35, 0.88, 0.425])


Q_vals_layer_0, Q_vals_layer_1 = generate_Q_map(object_pos, gripper_pos, goal_pos, subgoal_pos)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), )
plt.subplots_adjust(left=0.1, bottom=0.2)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
mpl.style.use('default')
number_of_layers_displayed = 2

im = [None] * 2

im[0] = axs[0].imshow(Q_vals_layer_0, interpolation="gaussian", cmap='viridis', vmin=-10,
                            vmax=0)
im[1] = axs[1].imshow(Q_vals_layer_1, interpolation="gaussian", cmap='viridis', vmin=-10,
                            vmax=0)

# Create tick labels
x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']

# Convert positions
object_pos_proj = proj_env2mpl(object_pos)
gripper_pos_proj = proj_env2mpl(gripper_pos)
goal_pos_proj = proj_env2mpl(goal_pos)
subgoal_pos_proj = proj_env2mpl(subgoal_pos)

# Generate patches for object, gripper, goal (and hole)
object = [None] * 2
gripper_1 = [None] * 2
gripper_2 = [None] * 2
goal = [None] * 2
hole = [None] * 2
cbar = [None] * 2

# Add everything to subplots
for i in range(2):
    object[i] = patches.Rectangle((object_pos_proj[0] - 1, object_pos_proj[1] - 1), 2, 2, alpha=0.7, linewidth=2,
                                  edgecolor='#000000', facecolor='#1a1a1a')
    gripper_1[i] = patches.Rectangle((gripper_pos_proj[0] - 1, gripper_pos_proj[1] - 1), 1, 2, alpha=0.7, linewidth=2,
                                  edgecolor='#000000', facecolor='#999999')
    gripper_2[i] = patches.Rectangle((gripper_pos_proj[0], gripper_pos_proj[1] - 1), 1, 2, alpha=0.7, linewidth=2,
                                  edgecolor='#000000', facecolor='#999999')
    goal[i] = patches.Rectangle((goal_pos_proj[0] - 1, goal_pos_proj[1] - 1), 2, 2, alpha=0.7, linewidth=2,
                             edgecolor='#000000', facecolor='#ffff00')
    if hparams["env"] == "FetchPush_variation1-v1":
        hole_pos_proj = proj_env2mpl(np.array([1.2, 0.675, 0.4]))
        hole[i] = patches.Rectangle((hole_pos_proj[0], hole_pos_proj[1]), 6, 8, alpha=0.5, linewidth=1, hatch='/',
                                 fill=False, edgecolor='#000000')
    elif hparams["env"] == "FetchPush_variation2-v1":
        hole_pos_proj = proj_env2mpl(np.array([1.25, 0.4, 0.4]))
        hole[i] = patches.Rectangle((hole_pos_proj[0], hole_pos_proj[1]), 14, 4, alpha=0.5, linewidth=1, hatch='/',
                                 fill=False, edgecolor='#000000')

    axs[i].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
    axs[i].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
    axs[i].set_xticklabels(x_label_list)
    axs[i].set_yticklabels(y_label_list)
    axs[i].set_title("Layer "+ str(i) + " (" + str(hparams["env"]) + ")")
    axs[i].add_patch(gripper_1[i])
    axs[i].add_patch(gripper_2[i])
    axs[i].add_patch(goal[i])
    if i == 0:
        subgoal = patches.Rectangle((subgoal_pos_proj[0] - 1, subgoal_pos_proj[1] - 1), 2, 2, alpha=0.7, linewidth=2,
                             edgecolor='#000000', facecolor='#ff00ff')
        axs[i].add_patch(subgoal)
    else:
        axs[i].add_patch(object[i])
    if hparams["env"] == "FetchPush_variation1-v1" or hparams["env"] == "FetchPush_variation2-v1":
        axs[i].add_patch(hole[i])

    # Create colorbars
    if i == 0:
        cbar[i] = axs[i].figure.colorbar(im[i], ax=axs, orientation="horizontal",
                                      boundaries=np.linspace(-10, 0, num=201), ticks=[-10, 0])
        cbar[i].ax.set_xlabel("Q-values", rotation=0, va="bottom")

        button_ax = plt.axes([0.1, 0.01, 0.25, 0.05])
        switch_disp_layers = Button(button_ax, 'Show only higher layer')
        switch_disp_layers.on_clicked(button_click)

shutil.rmtree('./temp_Q')
plt.show()
