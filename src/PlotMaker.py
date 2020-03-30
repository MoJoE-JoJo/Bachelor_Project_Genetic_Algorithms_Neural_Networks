import os.path
import sys
import csv

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Used to find modules when running from venv



input_file = open(sys.argv[1], "r")
input_reader = csv.DictReader(input_file)
data = [row for row in input_reader]
input_file.close()
path = ""

experiments = []
tempList = []
name = data[0]['name']
for line in data:
    if not(line['name'] == name):
        name = line['name']
        experiments.append(tempList)
        tempList = []
        tempList.append(line)
    else:
        tempList.append(line)

experiments.append(tempList)

experiments = list(filter(lambda x: not(x[0]['name'] == ''), experiments))

def make_plot_accuracy():
    plt.style.use('classic')
    fig, axes = plt.subplots(1, figsize=(12.5, 5))
    fig.suptitle('Average Accuracy')

    x=[]
    y=[]
    for exp in experiments:
        x.append([float(line['generation']) for line in exp if not(line['generation'] == '')])
        y.append([float(line['avg acc']) for line in exp if not(line['generation'] == '')])

    longest_x = []
    for list in x:
        if len(list) > len(longest_x):
            longest_x = list

    axes.set_xticks(longest_x)
    axes.set_xlim(min(longest_x) - 0.1, max(longest_x) + 0.1)
    axes.locator_params(axis='x', nbins=10)

    axes.xaxis.set_label_text('generation')
    axes.yaxis.set_label_text('accuracy')
    axes.label_outer()
    #axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    names = [exp[0]['name'] for exp in experiments]
    for i in range(0,len(x)):
        axes.plot(x[i],y[i], label=names[i], linewidth=1.2)

    fontP = FontProperties()
    fontP.set_size('small')
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)


    plt.savefig(fname=("plot_acc.svg"))

def make_plot_loss():
    plt.style.use('classic')
    fig, axes = plt.subplots(1, figsize=(12.5, 5))
    fig.suptitle('Average Loss')

    x=[]
    y=[]
    for exp in experiments:
        x.append([float(line['generation']) for line in exp if not(line['generation'] == '')])
        y.append([float(line['avg los']) for line in exp if not(line['generation'] == '')])

    longest_x = []
    for list in x:
        if len(list) > len(longest_x):
            longest_x = list

    axes.set_xticks(longest_x)
    axes.set_xlim(min(longest_x) - 0.1, max(longest_x) + 0.1)
    axes.locator_params(axis='x', nbins=10)

    axes.xaxis.set_label_text('generation')
    axes.yaxis.set_label_text('los')
    axes.label_outer()
    #axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    names = [exp[0]['name'] for exp in experiments]
    for i in range(0,len(x)):
        axes.plot(x[i],y[i], label=names[i], linewidth=1.2)

    fontP = FontProperties()
    fontP.set_size('small')
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)


    plt.savefig(fname=("plot_los.svg"))

def make_plot_parameters():
    plt.style.use('classic')
    fig, axes = plt.subplots(1, figsize=(12.5, 5))
    fig.suptitle('Average Amount of Parameters')

    x=[]
    y=[]
    for exp in experiments:
        x.append([float(line['generation']) for line in exp if not(line['generation'] == '')])
        y.append([float(line['avg par']) for line in exp if not(line['generation'] == '')])

    longest_x = []
    for list in x:
        if len(list) > len(longest_x):
            longest_x = list

    axes.set_xticks(longest_x)
    axes.set_xlim(min(longest_x) - 0.1, max(longest_x) + 0.1)
    axes.locator_params(axis='x', nbins=10)

    axes.xaxis.set_label_text('generation')
    axes.yaxis.set_label_text('parameters')
    axes.label_outer()
    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    names = [exp[0]['name'] for exp in experiments]
    for i in range(0,len(x)):
        axes.plot(x[i],y[i], label=names[i], linewidth=1.2)

    fontP = FontProperties()
    fontP.set_size('small')
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)


    plt.savefig(fname=("plot_par.svg"))

make_plot_accuracy()
make_plot_loss()
make_plot_parameters()