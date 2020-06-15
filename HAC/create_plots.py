import os
import shutil
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import iqr
import time




# Import arrays
datadir = Path("./data")
dates = []
date_paths = []
for x in datadir.iterdir():
    date_paths.append(x)
    dates.append(str(x)[5:])

mpl.style.use('default')
#mpl.style.use('seaborn')


assert len(dates) > 0, "No data found"


for j in range(len(date_paths)):
    current_path = date_paths[j]
    paths_to_graph = []

    for x in current_path.iterdir():
        if str(x)[-7:-1] == "graph_":
            paths_to_graph.append(x)
    f=open(str(paths_to_graph[0]) + "/title.txt", "r")  
    title =f.read()  

    plots = []
    plots_cl0 = []
    plots_cl1 = []

    for k in range(len(paths_to_graph)):
        current_path_to_graph = paths_to_graph[k]

        current_sr_list = []
        current_Q_val_list = []
        current_critic_loss_layer0_list = []
        current_critic_loss_layer1_list = []

        for x in current_path_to_graph.iterdir():
            if str(x)[-12:-5] == "sr_run_":
                current_sr_list.append(x)
            elif str(x)[-21:-5] == "Q_val_table_run_":
                current_Q_val_list.append(x)
            elif str(x)[-28:-5] == "critic_loss_layer0_run_":
                current_critic_loss_layer0_list.append(x)
            elif str(x)[-28:-5] == "critic_loss_layer1_run_":
                current_critic_loss_layer1_list.append(x)


            print("current_Q_val_list:", current_Q_val_list)



        #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
        #  Creating graphs for the average testing success rate  #
        #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
        test_graph = np.load(current_sr_list[0])

        current_sr_array = np.empty((len(current_sr_list), test_graph.shape[0]))

        for i in range(len(current_sr_list)):
            current_sr_array[i, :] = np.load(current_sr_list[i])     

        # plot only every n-th element
        n=1
        current_sr_array = current_sr_array[:,::n]
        current_sr_array = current_sr_array[:, :16]
        test_graph = test_graph[::n]
        test_graph = test_graph[:16]





        plots.append(current_sr_array)
        colors = [(0.3, 0.5, 0.9, 1.0), (0.2, 0.825, 0.2, 1.0), (1.0, 0.25, 0.25, 1.0), (0.7, 0.5, 0.85, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]
        label = True
        labels = ['ERB', 'transitions buffer']


        x = np.arange(0, test_graph.shape[0], 1)
        fig, ax = plt.subplots()

        # Calculate interquartile range
        intq_range = np.empty((len(plots), plots[0].shape[1]), dtype=float)
        for i in range(len(plots)):
            intq_range[i] = iqr(plots[i], axis=0)

        #print("intq_range:", intq_range)
        # Calculate average success rate
        average = np.empty((len(plots), plots[0].shape[1]), dtype=float)
        for i in range(len(plots)):
            average[i] = np.mean(plots[i], axis=0)
        #print("average:", average)

        yerr = intq_range

        for k in range(len(plots)):
            y = average[k]
            yerr = intq_range[k]
            if label:
                ax.plot(x, y, color=colors[k], label=labels[k])
                ax.legend(loc='lower right')
            else:
                ax.plot(x, y, color=colors[k])
            plt.fill_between(x, np.maximum(y-yerr, 0), np.minimum(y+yerr, 1), facecolor=colors[k], alpha=0.2)




    plt.title(title)
    plt.ylabel('Avg. Test Success Rate')
    ax.set_xlim((-0.5, plots[0].shape[1]-1 + 0.5))
    ax.set_ylim((-0.05, 1.05))
    plt.xlabel('Epoch')
    plt.grid(True)


    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    fig.set_size_inches(8, 4)
    Path("./figures").mkdir(parents=True, exist_ok=True)

    plt.savefig("./figures/" + "success_rate_plot_" + dates[j] + ".jpg", dpi=400, facecolor='w', edgecolor='w',
        orientation='landscape',transparent=False, bbox_inches='tight')

    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
    #                 Creating Q_val_figure                  #
    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
    Q_valsAndCritics=False

    if Q_valsAndCritics:
        # If Q-values are available
        if len(current_Q_val_list) > 0:
            # Loading Q-values from the first run
            first_Q_val_array = np.load(current_Q_val_list[0])
            # first_Q_val_array (step, layer (0,1), x-dim (10), y-dim (14))

            print("first_Q_val_array.shape:", first_Q_val_array.shape)
            num = np.ceil(np.sqrt(first_Q_val_array.shape[0])).astype(int)
            print("num:", num)

            # Q-values are plotted for a plane of the state space (orthogonal to z-axis)


            '''
            
                (1.05, 0.4)   --- y -->     (1.05, 1.1)

                    # # # # # # # # # # # # # #    |
                    #                         #    |
                    #                         # 
                    #                         #    x
                    #                         #    
                    #                         #    |
                    #                         #    V
                    #                         #
                    # # # # # # # # # # # # # #

                (1.55, 0.4)   --- y -->      (1.55, 1.1)



            '''

            # Layer 0
            if not np.array_equal(first_Q_val_array[0, 0, :, :], np.ones((20,28))):
                # Q_vals for the chosen steps have actually be written

                methods = ['gaussian']

                fig, axs = plt.subplots(nrows=num, ncols=num, figsize=(2*num, 2*num),)

                mpl.style.use('default')
                
                for l in range(first_Q_val_array.shape[0]):
                    im = axs.flat[l].imshow(first_Q_val_array[l, 0, :, :], interpolation="gaussian", cmap='viridis', vmin=-10, vmax=0)

                    # Create tick labels
                    x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
                    y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']
                    axs.flat[l].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
                    axs.flat[l].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
                    axs.flat[l].set_xticklabels(x_label_list)
                    axs.flat[l].set_yticklabels(y_label_list)

                    # Create colorbars
                    cbar = axs.flat[l].figure.colorbar(im, ax=axs.flat[l], orientation="horizontal", boundaries=np.linspace(-10.0, 0.0, num=201), ticks=[-10, 0])
                    cbar.ax.set_xlabel("Q-values", rotation=0, va="bottom")
                '''



                im = axs.flat[0].imshow(first_Q_val_array[0, 0, :, :], interpolation="gaussian", cmap='viridis', vmin=-10, vmax=0)

                # Create tick labels
                x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
                y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']
                axs.flat[0].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
                axs.flat[0].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
                axs.flat[0].set_xticklabels(x_label_list)
                axs.flat[0].set_yticklabels(y_label_list)

                # Create colorbars
                cbar = axs.flat[0].figure.colorbar(im, ax=axs.flat[0], orientation="horizontal", boundaries=np.linspace(-10.0, 0.0, num=201), ticks=[-10, -7.5, -5, -2.5, 0])
                cbar.ax.set_xlabel("Q-values", rotation=0, va="top")

                im = axs.flat[1].imshow(first_Q_val_array[60, 0, :, :], interpolation="gaussian", cmap='viridis', vmin=-10, vmax=0)

                # Create tick labels
                x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
                y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']
                axs.flat[1].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
                axs.flat[1].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
                axs.flat[1].set_xticklabels(x_label_list)
                axs.flat[1].set_yticklabels(y_label_list)

                # Create colorbars
                cbar = axs.flat[1].figure.colorbar(im, ax=axs.flat[1], orientation="horizontal", boundaries=np.linspace(-10.0, 0.0, num=201), ticks=[-10, -7.5, -5, -2.5, 0])
                cbar.ax.set_xlabel("Q-values", rotation=0, va="top")
                '''
                

                fig.tight_layout()
                Path("./figures").mkdir(parents=True, exist_ok=True)

                plt.savefig("./figures/" + dates[j] + "_Q_vals_layer_0.jpg", dpi=400, facecolor='w', edgecolor='w',
                    orientation='landscape',transparent=False, bbox_inches='tight')
                #plt.show()
                

            # Layer 1
            if not np.array_equal(first_Q_val_array[0, 1, :, :], np.ones((20,28))):
                # Q_vals for the choosen steps have actually be written

                methods = ['gaussian']

                fig, axs = plt.subplots(nrows=num, ncols=num, figsize=(2*num, 2*num),)

                mpl.style.use('default')
                
                for l in range(first_Q_val_array.shape[0]):
                    im = axs.flat[l].imshow(first_Q_val_array[l, 1, :, :], interpolation="gaussian", cmap='viridis', vmin=-7.5, vmax=-2.5)

                    # Create tick labels
                    x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
                    y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']
                    axs.flat[l].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
                    axs.flat[l].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
                    axs.flat[l].set_xticklabels(x_label_list)
                    axs.flat[l].set_yticklabels(y_label_list)

                    # Create colorbars
                    cbar = axs.flat[l].figure.colorbar(im, ax=axs.flat[l], orientation="horizontal", boundaries=np.linspace(-7.5, -2.5, num=201), ticks=[-7.5, -2.5])
                    cbar.ax.set_xlabel("Q-values", rotation=0, va="bottom")
                '''
                
                im = axs.flat[0].imshow(first_Q_val_array[0, 1, :, :], interpolation="gaussian", cmap='viridis', vmin=-7.5, vmax=-2.5)

                # Create tick labels
                x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
                y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']
                axs.flat[0].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
                axs.flat[0].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
                axs.flat[0].set_xticklabels(x_label_list)
                axs.flat[0].set_yticklabels(y_label_list)

                # Create colorbars
                cbar = axs.flat[0].figure.colorbar(im, ax=axs.flat[0], orientation="horizontal", boundaries=np.linspace(-7.5, -2.5, num=201), ticks=[-7.5, -5, -2.5])
                cbar.ax.set_xlabel("Q-values", rotation=0, va="top")

                im = axs.flat[1].imshow(first_Q_val_array[60, 1, :, :], interpolation="gaussian", cmap='viridis', vmin=-7.5, vmax=-2.5)

                # Create tick labels
                x_label_list = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
                y_label_list = ['1.05', '1.15', '1.25', '1.35', '1.45', '1.55']
                axs.flat[1].set_xticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5])
                axs.flat[1].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5, 19.5])
                axs.flat[1].set_xticklabels(x_label_list)
                axs.flat[1].set_yticklabels(y_label_list)

                # Create colorbars
                cbar = axs.flat[1].figure.colorbar(im, ax=axs.flat[1], orientation="horizontal", boundaries=np.linspace(-7.5, -2.5, num=201), ticks=[-7.5, -5, -2.5])
                cbar.ax.set_xlabel("Q-values", rotation=0, va="top")

                '''

                fig.tight_layout()
                Path("./figures").mkdir(parents=True, exist_ok=True)

                plt.savefig("./figures/" + dates[j] + "_Q_vals_layer_1.jpg", dpi=400, facecolor='w', edgecolor='w',
                    orientation='landscape',transparent=False, bbox_inches='tight')


        #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
        #               Creating critic_loss figure              #
        #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
        for k in range(len(paths_to_graph)):
            current_path_to_graph = paths_to_graph[k]

            current_sr_list = []
            current_Q_val_list = []
            current_critic_loss_layer0_list = []
            current_critic_loss_layer1_list = []

            for x in current_path_to_graph.iterdir():
                if str(x)[-12:-5] == "sr_run_":
                    current_sr_list.append(x)
                elif str(x)[-21:-5] == "Q_val_table_run_":
                    current_Q_val_list.append(x)
                elif str(x)[-28:-5] == "critic_loss_layer0_run_":
                    current_critic_loss_layer0_list.append(x)
                elif str(x)[-28:-5] == "critic_loss_layer1_run_":
                    current_critic_loss_layer1_list.append(x)

            print("current_critic_loss_layer0_list:", current_critic_loss_layer0_list)

            # layer 0
            if len(current_critic_loss_layer0_list) > 0:
                test_graph = np.load(current_critic_loss_layer0_list[0])

                current_critic_loss_layer0_array = np.empty((len(current_critic_loss_layer0_list), test_graph.shape[0]))

                for i in range(len(current_critic_loss_layer0_list)):
                    current_critic_loss_layer0_array[i, :] = np.load(current_critic_loss_layer0_list[i])

                plots_cl0.append(current_critic_loss_layer0_array)
                colors = [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.7, 0.5, 0.85, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]

                x = np.arange(0, test_graph.shape[0], 1)
                fig, ax = plt.subplots()

                # Calculate interquartile range
                intq_range = np.empty((len(plots_cl0), plots_cl0[0].shape[1]), dtype=float)
                for i in range(len(plots_cl0)):
                    intq_range[i] = iqr(plots_cl0[i], axis=0)

                #print("intq_range:", intq_range)
                # Calculate average success rate
                average = np.empty((len(plots_cl0), plots_cl0[0].shape[1]), dtype=float)
                for i in range(len(plots_cl0)):
                    average[i] = np.mean(plots_cl0[i], axis=0)
                #print("average:", average)

                yerr = intq_range


                for k in range(len(plots_cl0)):
                    y = average[k]
                    yerr = intq_range[k]
                    ax.plot(x, y, color=colors[k])
                    plt.fill_between(x, y-yerr, y+yerr, facecolor=colors[k], alpha=0.2)

                plt.ylabel('Critic loss')
                ax.set_xlim((0, plots_cl0[0].shape[1]-1))
                #ax.set_ylim((0.0, 1.2))
                plt.xlabel('Epoch')
                plt.grid(True)
                fig.set_size_inches(8, 4)
                Path("./figures").mkdir(parents=True, exist_ok=True)

                plt.savefig("./figures/" + "critic_loss_layer0_plot_" + dates[j] + ".jpg", dpi=400, facecolor='w', edgecolor='w',
                    orientation='landscape',transparent=False, bbox_inches='tight')

            # layer 1
            if len(current_critic_loss_layer1_list) > 0:
                test_graph = np.load(current_critic_loss_layer1_list[0])
                
                if test_graph[0] > -1:
                    current_critic_loss_layer1_array = np.empty((len(current_critic_loss_layer1_list), test_graph.shape[0]))

                    for i in range(len(current_critic_loss_layer1_list)):
                        current_critic_loss_layer1_array[i, :] = np.load(current_critic_loss_layer1_list[i])

                    plots_cl1.append(current_critic_loss_layer1_array)
                    colors = [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.7, 0.5, 0.85, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]

                    x = np.arange(0, test_graph.shape[0], 1)
                    fig, ax = plt.subplots()

                    # Calculate interquartile range
                    intq_range = np.empty((len(plots_cl1), plots_cl1[0].shape[1]), dtype=float)
                    for i in range(len(plots_cl1)):
                        intq_range[i] = iqr(plots_cl1[i], axis=0)

                    #print("intq_range:", intq_range)
                    # Calculate average success rate
                    average = np.empty((len(plots_cl1), plots_cl1[0].shape[1]), dtype=float)
                    for i in range(len(plots_cl1)):
                        average[i] = np.mean(plots_cl1[i], axis=0)
                    #print("average:", average)

                    yerr = intq_range


                    for k in range(len(plots_cl1)):
                        y = average[k]
                        yerr = intq_range[k]
                        ax.plot(x, y, color=colors[k])
                        plt.fill_between(x, y-yerr, y+yerr, facecolor=colors[k], alpha=0.2)

                    plt.ylabel('Critic loss')
                    ax.set_xlim((0, plots_cl1[0].shape[1]-1))
                    #ax.set_ylim((0.0, 1.2))
                    plt.xlabel('Epoch')
                    plt.grid(True)
                    fig.set_size_inches(8, 4)
                    Path("./figures").mkdir(parents=True, exist_ok=True)

                    plt.savefig("./figures/" + "critic_loss_layer1_plot_" + dates[j] + ".jpg", dpi=400, facecolor='w', edgecolor='w',
                        orientation='landscape',transparent=False, bbox_inches='tight')



