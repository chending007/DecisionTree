# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 可视化
import numpy as np
import pydotplus
from termcolor import colored as cl  # 文本自定义

from sklearn.tree import DecisionTreeClassifier as dtc  # 树算法
from sklearn import tree
from sklearn.model_selection import train_test_split  # 拆分数据
from sklearn.metrics import accuracy_score  # 模型准确度

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parameters:
    column_selection_condition = 1
    test_sample_size = 0.2
    tree_depth = 4
    tree_criterion = 'entropy'

    # 数据处理
    df = pd.read_csv('./TCGA-KIRC.mirna.tsv', delimiter='\t')
    df = df.T
    df.columns = df.iloc[0, :]
    df = df.iloc[1:len(df), :]
    df_column_mean = df.mean()
    df = df.drop(df_column_mean[df_column_mean < column_selection_condition].index, axis=1)
    ill = pd.DataFrame(data=df.index, columns=['ill_condition'])  # 向量形式
    ill_matrix = np.empty([len(df.index) - 1, 2], dtype='bool')
    x_vars = df[df.columns.to_list()].values
    for ill_index, ill_flag in enumerate(df.index):
        if ill_flag.split('-', 3)[3] != "11A":
            ill.iloc[[ill_index], 0] = True
        else:
            ill.iloc[[ill_index], 0] = False
    y_vars = ill['ill_condition'].values
    y_vars = y_vars.astype('bool')

    # 拆分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_vars, y_vars, test_size=test_sample_size, random_state=0)
    print("清洗后数据共计" + str(len(y_vars)) + "例，其中患者" + str(len(y_vars[y_vars == True])) + "例，非患者" + str(
        len(y_vars[y_vars == False])) + "例")
    print("测试集数据共计" + str(len(y_test)) + "例，其中患者" + str(len(y_test[y_test == True])) + "例，非患者" + str(
        len(y_test[y_test == False])) + "例")
    print("训练集数据共计" + str(len(y_train)) + "例，其中患者" + str(len(y_train[y_train == True])) + "例，非患者" + str(
        len(y_train[y_train == False])) + "例")
    print("测试集数据共计"+str(len(y_test))+"例，其中患者"+str(len(y_test[y_test == True]))+"例，非患者"+str(
        len(y_test[y_test == False]))+"例")
    model = dtc(criterion=tree_criterion, max_depth=tree_depth)
    model.fit(x_train, y_train)  # 训练数据
    pred_model = model.predict(x_test)
    print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs=['bold']))

    # 绘图
    feature_names = df.columns.to_list()[0:len(df.columns.to_list())]
    target_names = model.classes_
    if model.classes_[0]:
        target_names = ["patient", "health"]
    else:
        target_names = ["health", "patient"]

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, pred_model, labels=[True, False])  # True代表患者，False代表健康
    print("混淆矩阵：\n", cm)

    # 计算ROC 和 AUC
    from sklearn.metrics import roc_curve
    from sklearn import metrics
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(visible=True, ls=':')
    plt.title(u'DecisionTree ROC curve And AUC', fontsize=18)
    plt.show()

    # 一种绘图方式
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_pdf('tree.pdf')

    # 令一种绘图方式
    # plot_tree(model,
    #           feature_names=feature_names,
    #           class_names=["health", "patient"],
    #           filled=True,
    #           rounded=True)
    # plt.savefig('tree_visualization.png')
