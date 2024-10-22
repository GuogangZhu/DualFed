import time
import matplotlib.pyplot as plt
import csv

def save_para(args, log_name):
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    with open(log_name, "w") as f:
        f.write('Training Start Time: ' + now_time + '\n')
        args_dict = args.__dict__
        for arg, value in args_dict.items():
            f.write(arg + ': ' + str(value) + '\n')

def save_fig(metric_dict, y_label, fig_name):
    legend_lis = list(metric_dict.keys())
    plt.figure()
    for domain_idx in legend_lis:
        plt.plot(range(len(metric_dict[domain_idx])), metric_dict[domain_idx])
    plt.ylabel(y_label)
    plt.legend(legend_lis)
    plt.savefig(fig_name)

def save_csv(metric_dict, csv_name):
    head_list = list(metric_dict.keys())
    with open(csv_name, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(head_list)
        writer.writerow([metric_dict[head_list[j]] for j in range(len(head_list))])
        csvfile.close()

