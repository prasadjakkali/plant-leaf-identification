#import time
print("Initializing........")
print("")
#time.sleep(4)
#########333333333333333333333333333#######################################3333333333333333333#########################3333333333333333333333
import numpy as np
import pandas as pd
import os
import string
import wikipedia
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) #changing index changes voices but ony 0 and 1 are working here
from PIL import Image

dataset = pd.read_csv("Flavia_features.csv")
type(dataset)

maindir = r'D:\PLI\rakp'
ds_path = maindir + "\\Flavia leaves dataset"
img_files = os.listdir(ds_path)

breakpoints = [1001,1059,1060,1122,1552,1616,1123,1194,1195,1267,1268,1323,1324,1385,1386,1437,1497,1551,1438,1496,2001,2050,2051,2113,2114,2165,2166,2230,2231,2290,2291,2346,2347,2423,2424,2485,2486,2546,2547,2612,2616,2675,3001,3055,3056,3110,3111,3175,3176,3229,3230,3281,3282,3334,3335,3389,3390,3446,3447,3510,3511,3563,3566,3621]
target_list = []
for file in img_files:
    target_num = int(file.split(".")[0])
    flag = 0
    i = 0 
    for i in range(0,len(breakpoints),2):
        if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):
            flag = 1
            break
    if(flag==1):
        target = int((i/2))
        target_list.append(target)
        
y = np.array(target_list)
X = dataset.iloc[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
#metrics.accuracy_score(y_test, y_pred)
#print(metrics.classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]
svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
svm_clf.fit(X_train, y_train)
#svm_clf.best_params_
means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print('')

y_pred_svm = svm_clf.predict(X_test)
#metrics.accuracy_score(y_test, y_pred_svm)
#print(metrics.classification_report(y_test, y_pred_svm))

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
var= pca.explained_variance_ratio_

import matplotlib.pyplot as plt
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)####  
##################
####################################333333333333333333333333333###################################333333333333333333###############3333333333


import os
import cv2
def bg_sub(filename):
	test_img_path = '..\\mobile captures\\' + filename
	main_img = cv2.imread(test_img_path)

	img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
	resized_image = cv2.resize(img, (1600, 1200))
	size_y,size_x,_ = img.shape
	gs = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gs, (55,55),0)
	ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	kernel = np.ones((50,50),np.uint8)
	closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    
	contours = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

	contains = []
	y_ri,x_ri, _ = resized_image.shape
	for cc in contours:
		yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
		contains.append(yn)

	val = [contains.index(temp) for temp in contains if temp>0]
	index = val[0]
    
	black_img = np.empty([1200,1600,3],dtype=np.uint8)
	black_img.fill(0)
    
	cnt = contours[index]
	mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)
    
	maskedImg = cv2.bitwise_and(resized_image, mask)
	white_pix = [255,255,255]
	black_pix = [0,0,0]
    
	final_img = maskedImg
	h,w,channels = final_img.shape
	for x in range(0,w):
		for y in range(0,h):
			channels_xy = final_img[y,x]
			if all(channels_xy == black_pix):
				final_img[y,x] = white_pix
#	window_name = 'image'
#	cv2.imshow(window_name, main_img)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()

	#vv.v.imp
	plt.imshow(final_img)
	plt.title('matplotlib.pyplot:background removed',fontweight ="bold")
	plt.show()
	return final_img

	
import mahotas as mt
def feature_extract(img):
	names = ['area','perimeter','pysiological_length','pysiological_width','aspect_ratio','rectangularity','circularity', \
			'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b', \
			'contrast','correlation','inverse_difference_moments','entropy'
			]
	df = pd.DataFrame([], columns=names)

    #Preprocessing
	gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#	vv.v.imp
	plt.imshow(gs,cmap='Greys_r')
	plt.title('Converting image to grayscale',fontweight ="bold")	
	plt.show()
	
	blur = cv2.GaussianBlur(gs, (25,25),0)
	plt.imshow(blur,cmap='Greys_r')
	plt.title('Smoothing image using Guassian filter of size (25,25)',fontweight ="bold")	
	plt.show()

	ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	plt.imshow(im_bw_otsu,cmap='Greys_r')
	plt.title('Adaptive image thresholding using Otsu\'s thresholding method',fontweight ="bold")	
	plt.show()

	kernel = np.ones((50,50),np.uint8)
	closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
	plt.imshow(closing,cmap='Greys_r')
	plt.title('Closing of holes using Morphological Transformation',fontweight ="bold")	
	plt.show()	
##	vv.v.imp



#########FUCK NOW




    #Shape features
	contours = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	cnt = contours[0]
	M = cv2.moments(cnt)
	area = cv2.contourArea(cnt)
	perimeter = cv2.arcLength(cnt,True)
	x,y,w,h = cv2.boundingRect(cnt)
	aspect_ratio = float(w)/h
	rectangularity = w*h/area
	circularity = ((perimeter)**2)/area

    #Color features
	red_channel = img[:,:,0]
	green_channel = img[:,:,1]
	blue_channel = img[:,:,2]
	blue_channel[blue_channel == 255] = 0
	green_channel[green_channel == 255] = 0
	red_channel[red_channel == 255] = 0	
	##
	##
#	plt.imshow(red_channel,cmap="Greys_r")
#	plt.show()
#	plt.imshow(green_channel,cmap="Greys_r")
#	plt.show()
#	plt.imshow(blue_channel,cmap="Greys_r")
#	plt.show()								##	

	red_mean = np.mean(red_channel)
	green_mean = np.mean(green_channel)
	blue_mean = np.mean(blue_channel)

	red_std = np.std(red_channel)
	green_std = np.std(green_channel)
	blue_std = np.std(blue_channel)

	#Texture features
	textures = mt.features.haralick(gs)
	ht_mean = textures.mean(axis=0)
	contrast = ht_mean[1]
	correlation = ht_mean[2]
	inverse_diff_moments = ht_mean[4]
	entropy = ht_mean[8]

	vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity,\
			red_mean,green_mean,blue_mean,red_std,green_std,blue_std,\
			contrast,correlation,inverse_diff_moments,entropy
			]

	df_temp = pd.DataFrame([vector],columns=names)
	df = df.append(df_temp)	
	
	return df

while True :
	print("Enter file name :") 
	filename = str(input())
	filename = filename+'.jpg'
#	tip = '..\\mobile captures\\' + filename
#	img = Image.open(tip)
#	img.show()
    #filename = '1.jpg' 
    
	try:
		bg_rem_img = bg_sub(filename)
#		plt.imshow(bg_rem_img)
		#plt.show()

		features_of_img = feature_extract(bg_rem_img)
		#features_of_img
		
		scaled_features = sc_X.transform(features_of_img)
		#print(scaled_features)
		#y_pred_mobile = svm_clf.predict(features_of_img)
		y_pred_mobile = svm_clf.predict(scaled_features)
		print("")
		print('Variable value in CNarray :',y_pred_mobile[0])

		common_names = ['bamboo','Chinese horse chestnut','Anhui Barberry', \
						'Chinese redbud','true indigo','Japanese maple','Nanmu',' castor aralia', \
						'Chinese cinnamon','goldenrain tree','Big-fruited Holly','Japanese cheesewood', \
						'wintersweet','camphortree','Japan Arrowwood','sweet osmanthus','Cedrus deodara','ginkgo, maidenhair tree', \
						'Crape myrtle','oleander','yew plum pine','Japanese Flowering Cherry','Glossy Privet',\
						'Chinese Toon','peach','Lotus corniculatus','trident maple','Beales barberry','southern magnolia',\
						'Canadian poplar','Chinese tulip tree','tangerine'
						]
		answer =common_names[y_pred_mobile[0]]
		print('Name of plant :',answer)
		print('')
		result = wikipedia.summary(answer, sentences = 1)
		print(result)
		engine.say(result)
		engine.runAndWait()
		
				
	except:
		print("")
		print("Error Occured , >.<")
	
		
	print("")
	print("do you want to continue the search program :")
	ans = input()
	if ans == "n" or ans == "N" or ans == 'no' or ans == 'No':
		break