import os
import re
import cv2

LSA_DIR="../normal_experiments/datasets/lsa32x32/"
PH_DIR="../normal_experiments/datasets/ph32/"
TMP_DIR="lsa/"

if not os.path.exists(TMP_DIR):
	os.makedirs(TMP_DIR)

def create_dir(images_dir,name):
	png_images = [images_dir+f for f in os.listdir(images_dir) if re.search('png|PNG', f)]
	i = 0
	for f in png_images:
		img = cv2.imread(f)
		out = TMP_DIR+name+'/'+str(i)+'-'+f.split(os.sep)[-1].split("_")[0]+".jpg"
		cv2.imwrite(out, img)
		i += 1

print("Processing LSA dataset...")
subdirs = [x[0] for x in os.walk(LSA_DIR)]
for subdir in subdirs[1:]:
	print("Processing "+subdir)
	name = subdir.split("lsa32x32/")[1]
	os.makedirs(TMP_DIR+name)
	create_dir(subdir+'/',name)

print("Processing RWTH dataset...")
if not os.path.exists('ph/'):
	os.makedirs('ph/')

with open(PH_DIR+'3359-ph2014-MS-handshape-annotations.txt','r') as f:
	i = 0
	for line in f.readlines():
		l = line.split()
		img = cv2.imread(PH_DIR+l[0])
		out = 'ph/'+str(i)+"-"+l[1]+".jpg"
		cv2.imwrite(out, img)
		i += 1
