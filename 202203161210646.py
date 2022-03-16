import pandas as pd
import numpy as np
import scipy.stats as stat
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import os


# Data Preprocessing
# ######################################################################################################################
# 数据预处理
def data_preprocessing(df_begin):
    df = df_begin.groupby(df_begin.index).mean()  # 按样本排序去重
    df_filtered = df.loc[(df != 0).all(axis=1)]  # 过滤删除全空值
    return df_filtered


# 获取背景文件
def get_background(ref_data, param_pcc):
    df_background = pd.DataFrame(columns=['edge_index', 'Gene1', 'Gene2'])
    # 生成背景文件
    index = ref_data.index.tolist()
    ref_array = np.array(ref_data).astype(float)
    ref_pcc = np.corrcoef(ref_array)
    flag = 0
    for i in range(1, ref_pcc.shape[0]):
        for j in range(0, i, 1):
            if abs(ref_pcc[i][j]) >= param_pcc:
                flag += 1
                df_background = df_background.append(
                    pd.DataFrame({'edge_index': [flag], 'Gene1': [index[i]], 'Gene2': [index[j]]}))
                # print(index[i], index[j], ref_pcc[i][j])
    # print(50*"=")
    # print(flag)
    background_data = df_background.set_index(['edge_index'])
    return background_data


# SSN Calculating
# ######################################################################################################################
# 计算ssn_score得分（Z检验）
def ssn_score(deta, pcc, nn):
    if pcc == 1:
        pcc = 0.99999999
    if pcc == -1:
        pcc = -0.99999999
    z = deta / ((1 - pcc * pcc) / (nn - 1))
    return z


# SSN Plus Calculating
def ssn_plus_calculation(ref_data, per_data, bg_data):
    res_ssn = {}
    for i in per_data.columns:
        print("ssn_" + i)
        plus_df = pd.merge(ref_data, per_data[i], on='GeneName')

        for j in bg_data.index:
            gene1 = bg_data.T.loc['Gene1', j]
            gene2 = bg_data.T.loc['Gene2', j]
            edge = gene1 + '+' + gene2
            # print(gene1, type(gene2))
            r1 = stat.pearsonr(list(plus_df.T[gene1])[:-1], list(plus_df.T[gene2])[:-1])[0]
            r2 = stat.pearsonr(list(plus_df.T[gene1]), list(plus_df.T[gene2]))[0]
            r = r2 - r1

            z = ssn_score(r, r1, len(list(plus_df.T[gene1])[:-1]))
            pvalue = 1 - stat.norm.cdf(abs(z))
            if pvalue < 0.01:
                res_ssn.setdefault(i, {})[edge] = r
            else:
                res_ssn.setdefault(i, {})[edge] = 0

    df_res_ssn = pd.DataFrame()
    for i in res_ssn.keys():
        df_res_ssn = pd.concat([df_res_ssn, pd.DataFrame.from_dict(res_ssn[i], orient='index', columns=[i])], axis=1)
    return df_res_ssn


# SSN Minus Calculating
def ssn_minus_calculation(ref_data, per_data, bg_data):
    res_ssn = {}
    for i in per_data.columns:
        print("ssn_" + i)
        minus_df = ref_data
        minus_df = minus_df.drop(columns=i)

        for j in bg_data.index:
            gene1 = bg_data.T.loc['Gene1', j]
            gene2 = bg_data.T.loc['Gene2', j]
            edge = gene1 + '+' + gene2
            # print(gene1, type(gene2))
            r1 = stat.pearsonr(list(minus_df.T[gene1]), list(minus_df.T[gene2]))[0]
            r2 = stat.pearsonr(list(ref_data.T[gene1]), list(ref_data.T[gene2]))[0]
            r = r2 - r1

            z = ssn_score(r, r1, len(list(minus_df.T[gene1])))
            pvalue = 1 - stat.norm.cdf(abs(z))
            if pvalue < 0.01:
                res_ssn.setdefault(i, {})[edge] = r
            else:
                res_ssn.setdefault(i, {})[edge] = 0

    df_res_ssn = pd.DataFrame()
    for i in res_ssn.keys():
        df_res_ssn = pd.concat([df_res_ssn, pd.DataFrame.from_dict(res_ssn[i], orient='index', columns=[i])], axis=1)
    return df_res_ssn


# Distances Calculating
# ######################################################################################################################
def jaccard_distances_calculating(merge_file):
    merge_filtered = merge_file.loc[~(merge_file == 0).all(axis=1)]
    # 获取samples名称
    sample_names = merge_filtered.columns
    # 格式化文件
    merge_file[merge_filtered.values != 0] = 1
    # 计算jaccard距离
    net_df = pd.DataFrame()
    for i in sample_names:
        for j in sample_names:
            x = np.asarray(merge_filtered[i].tolist(), np.int32)
            y = np.asarray(merge_filtered[j].tolist(), np.int32)
            X = np.vstack([x, y])
            temp = pdist(X, 'jaccard')[0]
            net_df.loc[i, j] = temp
    return net_df


# 欧氏距离
def euclidean_distances_calculating(merge_file):
    merge_filtered = merge_file.loc[~(merge_file == 0).all(axis=1)]
    # 获取samples名称
    sample_names = merge_filtered.columns
    # 格式化文件
    merge_filtered[merge_filtered.values != 0] = 1
    # 计算euclidean距离
    net_df = pd.DataFrame()
    for i in sample_names:
        for j in sample_names:
            x = np.asarray(merge_filtered[i].tolist(), np.int32)
            y = np.asarray(merge_filtered[j].tolist(), np.int32)
            X = np.vstack([x, y])
            temp = pdist(X, 'euclidean')[0]
            net_df.loc[i, j] = temp
    return net_df


# Clustering Linkage
# ######################################################################################################################
# 聚类，并获取聚类索引名和矩阵
def clutering_linkage(df_distances):
    samples_names = df_distances.index
    samples = df_distances.values
    # 层次聚类分析，average：平均距离，类与类间所有pairs距离的平均
    mergings = sch.linkage(samples, method='average')
    return samples_names, mergings


# 绘制聚类树状图
def draw_clustering_graph(labels_file, matrix_file):
    labels = labels_file
    matrix = matrix_file
    plt.figure(figsize=(30, 10))
    Z1 = sch.dendrogram(matrix, labels=labels, leaf_rotation=90, leaf_font_size=10, orientation='top')
    labels_true = list(labels)
    labels_pred = list(Z1['ivl'])
    # print(labels_true)
    # print(labels_pred)
    plt.show()
    return labels_true, labels_pred


# 计算ARI系数
# ######################################################################################################################
def get_ARI_result(res_dendrogram):
    # 获取真实标签和预测标签
    labels_tr = res_dendrogram[0]
    labels_true = []
    for i in labels_tr:
        labels_true.append(int(i.split("_")[1]))
    print(labels_true)
    labels_pr = res_dendrogram[1]
    labels_pred = []
    for i in labels_pr:
        labels_pred.append(int(i.split("_")[1]))
    print(labels_pred)

    # 计算ARI
    ari = adjusted_rand_score(labels_true, labels_pred)
    print('ARI调整兰德系数为：%f' % (ari))


df_data = pd.read_csv("dataset_Buettner.csv", index_col="GeneName")  # 读取原始数据文件
df_ref_data = data_preprocessing(df_data)  # 参考样本数据
df_per_data = data_preprocessing(df_data)  # 干扰样本数据
df_bg_data = get_background(df_ref_data, 0.8)   # 获取背景数据

# SSN-Plus
df_res_ssn_plus_data = ssn_plus_calculation(df_ref_data, df_per_data, df_bg_data)  # SSN-Plus

df_distances_data_plus_euclidean = euclidean_distances_calculating(df_res_ssn_plus_data)  # euclidean
df_linkage_labels_plus_euclidean = clutering_linkage(df_distances_data_plus_euclidean)[0]
df_linkage_matrix_plus_euclidean = clutering_linkage(df_distances_data_plus_euclidean)[1]
get_ARI_result(draw_clustering_graph(df_linkage_labels_plus_euclidean, df_linkage_matrix_plus_euclidean))


# df_res_ssn_minus_data = ssn_minus_calculation(df_ref_data, df_per_data, df_bg_data)  # SSN-Minus
# df_distances_data_minus = euclidean_distances_calculating(df_res_ssn_minus_data)
# clutering_linkage(df_distances_data_minus)
