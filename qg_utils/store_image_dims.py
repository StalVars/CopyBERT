#!/bin/python


import sys
import os
from PIL import Image
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import glob


if len(sys.argv) != 2:
   print("Usage: python", sys.argv[0], "<image_folder>")
   sys.exit(1)

image_folder = sys.argv[1]
#image_files = os.listdir(image_folder)
image_files = glob.glob(image_folder+"*.png") + glob.glob(image_folder+"*.jpg") + glob.glob(image_folder+"*.jpeg") + glob.glob(image_folder+"*.tif")
print("Total image files:", len(image_files))
image_files = [ os.path.basename(image) for image in image_files ]
print(image_files)

#image_files = [ sys.argv[1]+"/"+f for f in image_files ]

num_procs=12


def split_list(list_f,num_procs):
   list_splits=[]
   split_len = len(list_f) // num_procs
   for i in range(0,len(list_f), split_len):
       if i == len(list_f)-1:
          till=len(list_f)
       else:
          till=i+split_len
       list_splits.append(list_f[i:till])
   return list_splits
         

def check_sizes(image_file_list):
   sizes=[]
   for f in tqdm(image_file_list):
         image_path=image_folder+"/"+f
         image=Image.open(image_path).convert("RGB")
         #sizes[f]=(image.width, image.height)
         sizes.append( (f, (image.width, image.height)) )
   return sizes
         
def get_sizes(f):
         image_path=image_folder+"/"+f
         image=Image.open(image_path).convert("RGB")
         return (f, (image.width, image.height))
 

list_splits_images = split_list(image_files,num_procs)
with Pool(num_procs) as pool:
   #sizes=pool.map(get_sizes, image_files)
   split_sizes=pool.map(check_sizes, list_splits_images)
   sizes = [ s for sub_list in split_sizes for s in sub_list ]


size_dict=dict()
for s in sizes:
   size_dict[s[0]]=s[1]

with open(image_folder+"/sizes.pkl","wb") as f:
     pickle.dump(size_dict,f)
