import tensorflow as tf
import numpy as np

def nn_layer(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]
    
    if is_output:
        weight_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        bias_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    else:
        # 1/sqrt(f)
        fan_in_init = 1 / num_prev_neurons ** 0.5
        weight_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)
        bias_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init) 

    weights = tf.compat.v1.get_variable("weights", shape, initializer=weight_init)
    biases = tf.compat.v1.get_variable("biases", [num_next_neurons], initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    relu = tf.nn.relu(dot)
    return relu


def _get_combinations(combinations):
    hparams = [{}] *len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"])*len(combinations["env"])

    for h in range(len(combinations["env"])):
        for i in range(len(combinations["ac_n"])):
            for j in range(len(combinations["sg_n"])):
                for k in range(len(combinations["replay_k"])):
                    for l in range(len(combinations["layers"])):
                        for m in range(len(combinations["use_target"])):
                            for n in range(len(combinations["sg_test_perc"])):
                                for o in range(len(combinations["buffer"])):
                                    for p in range(len(combinations["samp_str"])):
                                        for q in range(len(combinations["modules"])):
                                            for r in range(len(combinations["use_tl"])):
                                                for s in range(len(combinations["tl-mode"])):
                                                    hparams[h*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"]) + i*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"]) + j*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"]) + k*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"]) + l*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"]) + m*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"])*len(combinations["sg_test_perc"]) + n*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"])*len(combinations["buffer"]) + o*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"])*len(combinations["samp_str"]) + p*len(combinations["tl-mode"])*len(combinations["use_tl"])*len(combinations["modules"]) + q*len(combinations["tl-mode"])*len(combinations["use_tl"]) + r*len(combinations["tl-mode"]) + s ]  = {
                                                        "env"           : combinations["env"][h],
                                                        "ac_n"          : combinations["ac_n"][i],
                                                        "sg_n"          : combinations["sg_n"][j],
                                                        "replay_k"      : combinations["replay_k"][k],
                                                        "layers"        : combinations["layers"][l],
                                                        "use_target"    : combinations["use_target"][m],
                                                        "sg_test_perc"  : combinations["sg_test_perc"][n],
                                                        "buffer"        : combinations["buffer"][o],
                                                        "samp_str"      : combinations["samp_str"][p],
                                                        "modules"       : combinations["modules"][q],
                                                        "use_tl"        : combinations["use_tl"][r],
                                                        "tl-mode"       : combinations["tl-mode"][s]

                                                        }

    return hparams