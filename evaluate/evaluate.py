import numpy as np

def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))	# log2 以2为底 np.reciprocal 返回参数逐元素的倒数  index是顺序
	return 0



