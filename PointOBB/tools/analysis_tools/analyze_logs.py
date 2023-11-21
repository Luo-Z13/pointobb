import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    # 设置线条样式和颜色
    line_styles = ['-', '--', '-.', ':']
    # 修改为其他颜色，不使用蓝色
    # colors = ['#87CEFA', '#FFA500']
    colors = ['#1f78b4', '#ff7f0e']
    fig, ax = plt.subplots(figsize=(8, 6))

    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[0]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')

            if 'mAP' in metric:
                xs = np.arange(1, max(epochs) + 1)
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch', fontsize=18)
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o', linewidth=2, markevery=10)
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter', fontsize=18)

                # 使用 savgol_filter 进行平滑
                # ys_smoothed = savgol_filter(ys, window_length=5, polyorder=3)
                ys_smoothed = savgol_filter(ys, window_length=13, polyorder=2)

                plt.plot(
                    xs, ys_smoothed, label=legend[i * num_metrics + j],
                    color=colors[i % len(colors)],
                    # linewidth=2,
                    linewidth=4,
                    alpha=0.8
                )

                # 添加阴影效果
                # plt.fill_between(xs, ys_smoothed - 0.1, ys_smoothed + 0.1, color=colors[i % len(colors)], alpha=0.2)
                plt.fill_between(xs, ys_smoothed - 0.05, ys_smoothed + 0.05, color=colors[i % len(colors)], alpha=0.2)

            plt.legend(fontsize=19)
        if args.title is not None:
            plt.title(args.title)
        
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)

        # 修改背景颜色
        plt.gca().set_facecolor('#f0f0f0')  # 修改为浅灰色背景

        # 添加背景和边框
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


# def plot_curve(log_dicts, args):
#     if args.backend is not None:
#         plt.switch_backend(args.backend)
#     sns.set_style(args.style)

#     # 设置线条样式和颜色
#     line_styles = ['-', '--', '-.', ':']
#     # colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 修改为其他颜色，不使用蓝色
#     colors = ['#87CEFA', '#FFA500']
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # if legend is None, use {filename}_{key} as legend
#     legend = args.legend
#     if legend is None:
#         legend = []
#         for json_log in args.json_logs:
#             for metric in args.keys:
#                 legend.append(f'{json_log}_{metric}')
#     assert len(legend) == (len(args.json_logs) * len(args.keys))
#     metrics = args.keys

#     num_metrics = len(metrics)
#     for i, log_dict in enumerate(log_dicts):
#         epochs = list(log_dict.keys())
#         for j, metric in enumerate(metrics):
#             print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
#             if metric not in log_dict[epochs[0]]:
#                 raise KeyError(
#                     f'{args.json_logs[i]} does not contain metric {metric}')

#             if 'mAP' in metric:
#                 xs = np.arange(1, max(epochs) + 1)
#                 ys = []
#                 for epoch in epochs:
#                     ys += log_dict[epoch][metric]
#                 ax = plt.gca()
#                 ax.set_xticks(xs)
#                 plt.xlabel('epoch', fontsize=18)
#                 plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
#             else:
#                 xs = []
#                 ys = []
#                 num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
#                 for epoch in epochs:
#                     iters = log_dict[epoch]['iter']
#                     if log_dict[epoch]['mode'][-1] == 'val':
#                         iters = iters[:-1]
#                     xs.append(
#                         np.array(iters) + (epoch - 1) * num_iters_per_epoch)
#                     ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
#                 xs = np.concatenate(xs)
#                 ys = np.concatenate(ys)
#                 plt.xlabel('iter', fontsize=18)
#                 # plt.plot(
#                 #     xs, ys, label=legend[i * num_metrics + j], linewidth=1)
#                 plt.plot(
#                     xs, ys, label=legend[i * num_metrics + j],
#                     color=colors[i % len(colors)],
#                     linewidth=2)
                
#             plt.legend(fontsize=18)
#         if args.title is not None:
#             plt.title(args.title)
        
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)

#         #TODO: 修改背景颜色
#         plt.gca().set_facecolor('#f0f0f0')  # 修改为浅灰色背景

#         # 添加背景和边框
#         plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     if args.out is None:
#         plt.show()
#     else:
#         print(f'save curve to: {args.out}')
#         plt.savefig(args.out)
#         plt.cla()


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


# def plot_curve_ori(log_dicts, args):
#     if args.backend is not None:
#         plt.switch_backend(args.backend)
#     sns.set_style(args.style)
#     # if legend is None, use {filename}_{key} as legend
#     legend = args.legend
#     if legend is None:
#         legend = []
#         for json_log in args.json_logs:
#             for metric in args.keys:
#                 legend.append(f'{json_log}_{metric}')
#     assert len(legend) == (len(args.json_logs) * len(args.keys))
#     metrics = args.keys

#     num_metrics = len(metrics)
#     for i, log_dict in enumerate(log_dicts):
#         epochs = list(log_dict.keys())
#         # TODO:仅绘制前4个 epoch 的数据
#         # epochs_to_plot = min(4, len(epochs))
#         epochs_to_plot = 4
#         for j, metric in enumerate(metrics):
#             print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
#             if metric not in log_dict[epochs[0]]:
#                 raise KeyError(
#                     f'{args.json_logs[i]} does not contain metric {metric}')

#             if 'mAP' in metric:
#                 xs = np.arange(1, epochs_to_plot + 1)  # 仅限前5个 epoch
#                 ys = []
#                 for epoch in epochs[:epochs_to_plot]:
#                     # 仅限前5个 epoch 的数据
#                     ys += log_dict[epoch][metric][:args.max_iters]
#                 ax = plt.gca()
#                 ax.set_xticks(xs)
#                 plt.xticks(fontsize=24)
#                 plt.yticks(fontsize=24)

#                 plt.xlabel('epoch')
#                 plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
#             else:
#                 xs = []
#                 ys = []
#                 num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
#                 for epoch in epochs[:epochs_to_plot]:
#                     iters = log_dict[epoch]['iter']
#                     if log_dict[epoch]['mode'][-1] == 'val':
#                         iters = iters[:-1]
#                     xs.append(
#                         np.array(iters) + (epoch - 1) * num_iters_per_epoch)
#                     ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
#                 xs = np.concatenate(xs)
#                 ys = np.concatenate(ys)
#                 plt.xticks(fontsize=24)
#                 plt.yticks(fontsize=24)
#                 plt.xlabel('iter')
#                 plt.plot(
#                     xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
#             plt.legend()
#         if args.title is not None:
#             plt.title(args.title)
        
#         plt.xticks(fontsize=24)
#         plt.yticks(fontsize=24)
#     if args.out is None:
#         plt.show()
#     else:
#         print(f'save curve to: {args.out}')
#         plt.savefig(args.out)
#         plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()


# def plot_curve(log_dicts, args):
#     if args.backend is not None:
#         plt.switch_backend(args.backend)
#     sns.set_style(args.style)
#     # if legend is None, use {filename}_{key} as legend
#     legend = args.legend
#     if legend is None:
#         legend = []
#         for json_log in args.json_logs:
#             for metric in args.keys:
#                 legend.append(f'{json_log}_{metric}')
#     assert len(legend) == (len(args.json_logs) * len(args.keys))
#     metrics = args.keys

#     num_metrics = len(metrics)
#     for i, log_dict in enumerate(log_dicts):
#         epochs = list(log_dict.keys())
#         for j, metric in enumerate(metrics):
#             print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
#             if metric not in log_dict[epochs[0]]:
#                 raise KeyError(
#                     f'{args.json_logs[i]} does not contain metric {metric}')

#             if 'mAP' in metric:
#                 xs = np.arange(1, max(epochs) + 1)
#                 ys = []
#                 for epoch in epochs:
#                     ys += log_dict[epoch][metric]
#                 ax = plt.gca()
#                 ax.set_xticks(xs)
#                 plt.xlabel('epoch')
#                 plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
#             else:
#                 xs = []
#                 ys = []
#                 num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
#                 for epoch in epochs:
#                     iters = log_dict[epoch]['iter']
#                     if log_dict[epoch]['mode'][-1] == 'val':
#                         iters = iters[:-1]
#                     xs.append(
#                         np.array(iters) + (epoch - 1) * num_iters_per_epoch)
#                     ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
#                 xs = np.concatenate(xs)
#                 ys = np.concatenate(ys)
#                 plt.xlabel('iter')
#                 plt.plot(
#                     xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
#             plt.legend()
#         if args.title is not None:
#             plt.title(args.title)
#     if args.out is None:
#         plt.show()
#     else:
#         print(f'save curve to: {args.out}')
#         plt.savefig(args.out)
#         plt.cla()