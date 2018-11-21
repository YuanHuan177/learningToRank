# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
reload(sys)
sys.setdefaultencoding("utf-8")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

######################################################################################
# 错误绘图函数。 当您不希望它被覆盖时更改名称, 因为它会自动保存

def acc(train_acc, test_acc, savename='result_acc.pdf'):
    ep = np.arange(len(train_acc)) + 1

    import matplotlib.pyplot as plt
    plt.switch_backend('agg')  #指定不需要GUI的backend

    plt.plot(ep, train_acc, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_acc, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title("Accuracy")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.savefig(savename)
    

    
def loss(train_loss, test_loss, savename='result_loss.pdf'):
    ep = np.arange(len(train_loss)) + 1

    import matplotlib.pyplot as plt
    plt.switch_backend('agg')  # 指定不需要GUI的backend

    plt.plot(ep, train_loss, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_loss, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title("Loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.legend(loc='upper right')
    plt.savefig(savename)
    
    
    