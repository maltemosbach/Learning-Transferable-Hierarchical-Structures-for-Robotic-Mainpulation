import os

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

data_folder = "data"
plots_folder = "plots"

plt.rc("lines", linewidth=2)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("axes", labelsize=16, titlesize=18)
plt.rc("legend", fontsize=16)

for scenario in os.listdir(data_folder):

    fig, ax = plt.subplots()
    ax.set_title(scenario)

    scenario_folder = os.path.join(data_folder, scenario)

    for variation in os.listdir(scenario_folder):

        variation_folder = os.path.join(scenario_folder, variation)

        all_data = []

        min_length = np.inf

        for run in os.listdir(variation_folder):
            if run.endswith(".npy"):
                run_file = os.path.join(variation_folder, run)
                data = np.load(run_file)
                all_data.append(data)
                # plt.plot(data, c="grey")
                min_length = min(min_length, len(data))

        all_data = [data[:min_length] for data in all_data]

        if all_data:
            all_data = np.stack(all_data)

            all_data *= 100

            mean = np.median(all_data, axis=0)
            top = np.quantile(all_data, .75, axis=0)
            bottom = np.quantile(all_data, .25, axis=0)

            episodes = np.arange(len(mean)) / 10

            ax.fill_between(episodes, bottom, top, alpha=.3)

            ax.plot(episodes, mean, label=str(variation))

            # ax.xaxis.set_major_formatter(lambda x, pos: x[:-2])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: str(x)[:-2] if str(x).endswith(".0") else str(x)))
            plt.yticks(np.arange(0, 101, 20))
            # ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%0.1f'))
            # ax.yaxis.set_ticklabels(range(100, 20))
            # plt.set

    plt.legend()
    plt.xlabel("Episodes [x1000]")
    plt.ylabel("Success Rate [%]")
    
    plt.tight_layout()

    plt.savefig(os.path.join(plots_folder, scenario + ".png"), dpi=300)

    plt.legend()
    plt.show()
    # os.path.join("../data")
