from scipy.stats import wilcoxon


def doWilcoxonTest(data1, data2, idealpValue):
    res_1_to_2 = wilcoxon(data1, data2, zero_method='wilcox', alternative='less')
    statistics_1_to_2, pvalue_1_to_2 = res_1_to_2.statistic, res_1_to_2.pvalue

    equal = -1
    better_1 = -1
    better_2 = -1

    if pvalue_1_to_2 == None:
        equal = 1
    elif pvalue_1_to_2 >= idealpValue:
        res_2_to_1 = wilcoxon(data2, data1, zero_method='wilcox', alternative='less')
        statistics_2_to_1, pvalue_2_to_1 = res_2_to_1.statistic, res_2_to_1.pvalue
        if pvalue_2_to_1 < idealpValue:
            better_2 = 1
        else:
            equal = 1
    else:
        better_1 = 1

    if equal == 1:
        return 0
    elif better_1 == 1:
        return 1
    elif better_2 == 1:
        return 2
    else:
        return -1


# if __name__ == '__main__':
#
