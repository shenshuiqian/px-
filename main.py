import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import hello


import re
import time
import cv2
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from tqdm import tqdm
from paddlenlp import Taskflow
sns.set_theme(style="ticks")
font = "C:\Windows\Fonts\simsun.ttc"
fp = fm.FontProperties(fname=font)
plt.rcParams["axes.unicode_minus"] = False
labels = ["Y", "I"]
filePath = "D:/msg.csv"
dStart = "2022-01-01 00:00:00"
dEnd = "2024-3-29 23:59:59"



df = pd.read_csv(filePath, encoding="utf-8")
df = df.query(
    "CreateTime >= {:d} and CreateTime <= {:d}".format(
        int(time.mktime(time.strptime(dStart, "%Y-%m-%d %H:%M:%S"))),
        int(time.mktime(time.strptime(dEnd, "%Y-%m-%d %H:%M:%S"))),
    )
)

df.loc[:, "StrTime"] = pd.to_datetime(df["StrTime"])
df.loc[:, "day"] = df["StrTime"].dt.dayofweek
df.loc[:, "hour"] = df["StrTime"].dt.hour
df.loc[:, "Count"] = 1


dfs = [df.query("IsSender == 0"), df.query("IsSender == 1")]

def textFilter(text: str):
    text = text.lower()
    # 过滤 emoji
    try:
        co = re.compile("[\U00010000-\U0010ffff]")
    except re.error:
        co = re.compile("[\uD800-\uDBFF][\uDC00-\uDFFF]")
    text = co.sub(" ", text)
    # 过滤微信表情
    co = re.compile("\[[\u4e00-\u9fa5]+\]")
    return co.sub(" ", text)




def function1():
    texts = [
        [textFilter(i) for i in dfs[0].query("Type == 1")["StrContent"].to_list()],
        [textFilter(i) for i in dfs[1].query("Type == 1")["StrContent"].to_list()],
    ]

    data = {}
    for i in range(2):
        data[labels[i]] = [
            len(dfs[i].query("Type == 1")),
            len(dfs[i].query("Type == 3")),
            len(dfs[i].query("Type == 34")),
            len(dfs[i].query("Type == 43")),
            len(dfs[i].query("Type == 47")),
        ]

    data = (
        pd.DataFrame(data, index=["Text", "Image", "Voice", "Video", "Sticker"])
        .reset_index()
        .melt("index")
        .rename(columns={"index": "Type", "variable": "Person", "value": "Count"})
    )
    my_palette={labels[0]:"red",labels[1]:"yellow"}
    g = sns.catplot(data, kind="bar", x="Type", y="Count", hue="Person", palette=my_palette, alpha=0.6, height=6)

    for ax in g.axes.ravel():
        for i in range(2):
            ax.bar_label(ax.containers[i], fontsize=9)
    sns.move_legend(g, "upper right")
    plt.yscale("log")
    g.figure.set_size_inches(6, 5)
    g.figure.set_dpi(150)

    plt.savefig('D:\\1.png')
    plt.close()
    img_path="D:\\1.png"
    img=cv2.imread(img_path)
    cv2.namedWindow('pic', 0)
    cv2.resizeWindow('pic',1200,900)
    cv2.imshow('pic', img)



def function2():
    ##聊天时间的频率
    multiple = "dodge"
    data = {"Time": [], "Person": []}
    for i in range(2):
        hour = dfs[i]["hour"].to_list()
        data["Time"] += hour
        data["Person"] += [labels[i]] * len(hour)

    data = pd.DataFrame(data)
    bins = np.arange(0, 25, 1)
    my_palette = {labels[0]: "red", labels[1]: "yellow"}
    ax = sns.histplot(
        data=data,
        x="Time",
        hue="Person",
        bins=bins,
        multiple=multiple,
        edgecolor=".3",
        linewidth=0.5,
        palette=my_palette,
        alpha=0.6,
    )
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Hour")
    ax.set_xlim(0, 24)
    sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)

    ax.figure.set_size_inches(8, 4)
    ax.figure.set_dpi(150)
    plt.savefig('D:\\1.png')
    plt.close()
    img_path = "D:\\1.png"
    img = cv2.imread(img_path)
    cv2.namedWindow('pic', 0)
    cv2.resizeWindow('pic', 1400, 700)
    cv2.imshow('pic', img)

def function3():
    ##情感分析
    dfE = df.query("Type == 1")[["IsSender", "StrContent", "StrTime", "hour"]]
    dfE.index = range(len(dfE))

    senta = Taskflow("sentiment_analysis")
    scores = pd.DataFrame(senta([textFilter(i) for i in dfE["StrContent"].to_list()]))
    scores.loc[scores["label"] == "negative", "score"] = 1 - scores.loc[scores["label"] == "negative", "score"]

    dfE["score"] = scores["score"]
    dfE["score"] = 2 * dfE["score"] - 1
    dfE["Person"] = dfE.apply(lambda x: labels[x["IsSender"]], axis=1)

    dfEs = [dfE.query("IsSender == 0"), dfE.query("IsSender == 1")]
    my_palette = {labels[0]: "red", labels[1]: "blue"}
    ax = sns.histplot(data=dfE, x="score", hue="Person", palette=my_palette, alpha=0.6, bins=100)

    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    ax.set_xlim(-1, 1)

    ax.figure.set_size_inches(8, 3)
    ax.figure.set_dpi(150)
    plt.savefig('D:\\1.png')
    plt.close()
    img_path = "D:\\1.png"
    img = cv2.imread(img_path)
    cv2.namedWindow('pic', 0)
    cv2.resizeWindow('pic', 1400, 700)
    cv2.imshow('pic', img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = hello.Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.pushButton.clicked.connect(function1)
    ui.pushButton_2.clicked.connect(function2)
    ui.pushButton_3.clicked.connect(function3)
    MainWindow.show()
    sys.exit(app.exec_())