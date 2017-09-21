import os
import re
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--koller_dir", required=True,
	help="path to the directory of koller dataset")
args = vars(ap.parse_args())

images_dir = os.path.join(args['koller_dir'],'')

if not os.path.exists('koller/'):
	os.makedirs('koller/')

with open(images_dir+'3359-ph2014-MS-handshape-annotations.txt','r') as f:
	i = 0
	for line in f.readlines():
		l = line.split()
		print(l[0])
		img = cv2.imread(images_dir+l[0])
		out = 'koller/'+str(i)+"-"+l[1]+".jpg"
		cv2.imwrite(out, img)
		i += 1
