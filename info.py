import os
import shutil
from random import shuffle

src_dir = '/var/home/l_ltindall/plankton_images'


src_dirs = os.listdir(src_dir+'/all_images')
print src_dirs
total_files = 0
for cat in src_dirs: 
    if os.path.isdir(src_dir+'/all_images/'+cat): 
        print 'files in ',cat

        files = os.listdir(src_dir+'/all_images/'+cat)
        print len(files)
        total_files = total_files + len(files)
   
	#print files[:10]
        #print "vs."
        #print files[:10]
	
        

print total_files
'''
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)

'''
