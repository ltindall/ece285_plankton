'''
This script should be placed in a directory which contains 
an all_images directory. In all_images there should be subdirectories 
with names according to classification labels. 

There should also be an empty keras_images directory. This directory 
will be populated with a train and test directory each filled with 
subdirectories according to classification labels. 

These train and test directories can be passed into Keras'
ImageDataGenerator.flow_from_directory method. 

Usage: 
    python split.py


'''



import os
import shutil
from random import shuffle



src_dir = '/var/home/l_ltindall/plankton_images'

if not os.path.exists(src_dir+'/keras_images/train'):
    os.makedirs(src_dir+'/keras_images/train')

if not os.path.exists(src_dir+'/keras_images/test'):
    os.makedirs(src_dir+'/keras_images/test')

src_dirs = os.listdir(src_dir+'/all_images')
print src_dirs
total_files = 0
for cat in src_dirs: 
    if os.path.isdir(src_dir+'/all_images/'+cat): 
        #print 'files in ',cat

        files = os.listdir(src_dir+'/all_images/'+cat)
        #print len(files)
        total_files = total_files + len(files)
   
	#print files[:10]
        shuffle(files)
        #print "vs."
        #print files[:10]
	
	os.makedirs(src_dir+'/keras_images/train/'+cat)
        os.makedirs(src_dir+'/keras_images/test/'+cat)
        
        i = 0
        for f in files: 
            if i >230: 
                shutil.copy(src_dir+'/all_images/'+cat+'/'+f,src_dir+'/keras_images/train/'+cat+'/'+f)
            else: 
                shutil.copy(src_dir+'/all_images/'+cat+'/'+f,src_dir+'/keras_images/test/'+cat+'/'+f)
            i = i + 1
        #print os.listdir(src_dir+'/'+cat)
        

print total_files
'''
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)

'''
