import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder = './logs'
    linear_probing_log = {}

    file_names = ['linear_mnist', 'linear_cifar10', 'linear_stl10', 'linear_cifar20']

    for file_name in file_names:

        file_name = file_name + '.log'

        dset_name = file_name[7:-4]
        Top1 = []
        Top3 = []

        cur_top1 = []
        cur_top3 = []

        with open(os.path.join(folder, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                if 'Top1' in line:
                    top1_score = float(line[line.index(':')+2:line.index(':')+9])
                    top3_score = float(line[line.rindex(':')+2:line.rindex(':')+9])
                    cur_top1.append(top1_score)
                    cur_top3.append(top3_score)

                    if len(cur_top1) == 10:
                        Top1.append(max(cur_top1))
                        Top3.append(max(cur_top3))

                        cur_top1 = []
                        cur_top3 = []

        linear_probing_log[dset_name] = [Top1, Top3]

        Top1 = []
        Top3 = []


    color = ['r', 'b', 'g', 'm']
    markers = ['*', 'o', 'D', 'x']
    handles = []
    x_axis = list(range(100, 900, 100))
    for idx, (dset_name, (Top1, Top3)) in enumerate(linear_probing_log.items()):

        handles.append(plt.plot(x_axis, Top1, color=color[idx], marker=markers[idx], label=dset_name, linewidth=2.5))

    print(list(linear_probing_log.keys()))
    plt.grid(ls='--')
    plt.xlabel('epochs')
    plt.ylabel('Top-1 accuracy')
    plt.legend(loc='upper right')
    plt.savefig('./imgs/linear_probing_top1.png')

