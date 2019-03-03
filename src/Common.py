def make_dic_of_cluster_result(cluster_result):
    """
    根据分类结果，返回item_index到对应聚类的字典索引
    :param cluster_result:
    :return:
    """
    cluster_result_dic = {};

    for cluster_index in range(cluster_result.__len__()):
        for index in cluster_result[cluster_index] :
            cluster_result_dic[index] = cluster_index;

    return cluster_result_dic;