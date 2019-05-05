import os
from PIL import Image
import numpy as np
 
def tobin(data):
	row = data.shape[1]
	col = data.shape[2]
	bindata = np.empty(row * col)
	for i in range(row):
		for j in range(col):
			bindata[i * col + j] = 0
			if(data[0][i][j] > 127):
				bindata[i * col + j] = 1
	return bindata
 
def load_data(data_path, split):
	files = os.listdir(data_path)
	file_num = len(files)
	index = np.random.permutation(file_num)
	selected_file_num = 40000
	selected_files = []
	for i in range(selected_file_num):
		selected_files.append(files[index[i]])
 
	img_matrix = np.empty((selected_file_num, 1, 28, 28), dtype = "float32")
	data = np.empty((selected_file_num, 28 * 28), dtype = "float32")
	label = np.empty((selected_file_num), dtype = "uint8")

	filenames=[]
 
	print "loading data..."
	for i in range(selected_file_num):
		#print i,"/",selected_file_num,"\r",
		file_name = selected_files[i]
		filenames.append(file_name)
		file_path = os.path.join(data_path, file_name)
		img_matrix[i] = Image.open(file_path)
		data[i] = tobin(img_matrix[i])
		label[i] = int(file_name.split('.')[0])
	print ""
 
	division = (int)(split * selected_file_num)
	index = np.random.permutation(selected_file_num)
	train_index, test_index = index[:division], index[division:]
	train_label, test_label = label[train_index], label[test_index]
	train_data, test_data = data[train_index], data[test_index]
	

	train_name=[]
	test_name=[]
	for i in range(len(train_index)):
		train_name.append(filenames[train_index[i]])
	for i in range(len(test_index)):
		test_name.append(filenames[test_index[i]])
	
	return train_data, train_label, test_data, test_label, train_name, test_name
 
def KNN(test_name, test_label, test_data, train_data, train_label, train_name, k):
	train_data_size = train_data.shape[0]
	dif_matrix = np.tile(test_data, (train_data_size, 1)) - train_data
	euc_dif_matrix = dif_matrix ** 2
	euc_dis_vec = euc_dif_matrix.sum(axis = 1)
	sorted_index = euc_dis_vec.argsort()
 
	max_count = 0
	best_class = 0
	best_class_s=[]
	class_count = {}
	

	print "true label: ", test_label
	print "true name: ", test_name
	ifprint=0

	for j in range(1,k+1):
		if j==10:
			ifprint=1
		for i in range(j):
			this_tclass = train_label[sorted_index[i]]
			if ifprint==1:
				print train_name[sorted_index[i]],
			this_tclass_count = class_count.get(this_tclass, 0) + 1
			class_count[this_tclass] = this_tclass_count
			if(this_tclass_count > max_count):
				max_count = this_tclass_count
				best_class = this_tclass
		if ifprint==1:
			print ""
		best_class_s.append(best_class)
		print "k=",j,"best class: ",best_class
		class_count.clear()
		max_count=0
		best_class=0
	print ""

	return best_class_s
 
if __name__=="__main__":
	np.random.seed(333333)
	span=10
	train_data, train_label, test_data, test_label, train_name, test_name= load_data("../mnist_data", 0.95)
	total = test_data.shape[0]
	err = [0.0]*span
	print "testing..."
	for i in range(total):
		print i+1,"/",total
		best_class_s = KNN(test_name[i], test_label[i], test_data[i], train_data, train_label, train_name,span)
		print "\n"
		for j in range(len(best_class_s)):
			if(best_class_s[j] != test_label[i]):
				err[j] = err[j] + 1.0
	print ""
	print "misclassification rate"
	for i in range(span):
		print "k=",i+1,":",err[i] / total," ",err[i]
	
