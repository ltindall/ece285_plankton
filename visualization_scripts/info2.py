import os
import shutil
from random import shuffle

src_dir = '/var/home/l_ltindall/plankton_images'


src_dirs = os.listdir(src_dir+'/all_images')
print src_dirs
total_files = 0
train_files = 0
test_files = 0
for cat in src_dirs: 
    if os.path.isdir(src_dir+'/all_images/'+cat): 
        #print 'files in ',cat

        files = os.listdir(src_dir+'/all_images/'+cat)
        #print len(files)
        total_files = total_files + len(files)
   
	
        
        i = 0
        for f in files: 
            if i >230: 
                train_files = train_files + 1
            else: 
                test_files = test_files + 1
            i = i + 1
        #print os.listdir(src_dir+'/'+cat)
        

print 'total files = ',total_files
print 'train files = ',train_files
print 'test files = ',test_files
'''
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)

'''
