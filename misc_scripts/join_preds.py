#!/bin/python

import glob
import sys
from natsort import natsorted
import os
import re


files=glob.glob(sys.argv[1]+"*")
ref_files=glob.glob(sys.argv[2]+"*")

files=natsorted(files)
ref_files=natsorted(ref_files)


def postprocess(line):
    line=line.strip()
    line=re.sub(" ##","",line)
    return line


preds=[]
print("Pred:")
for fil in files:
    with open(fil,"r") as f:
      for pred in f:
         pred=postprocess(pred)
         preds.append(pred)
      print(fil, len(preds))



refs=[]
print("Ref:")
for fil in ref_files:
    with open(fil,"r") as f:
      for ref in f:
         ref=postprocess(ref)
         refs.append(ref)
      print(fil,len(refs))


print("all preds:", len(preds))
print("all refs:", len(refs))
outputdir=os.path.dirname(files[0])
with open(outputdir+"/preds_all.txt","w") as f:
    for pred in preds:
        f.write(pred+"\n")
        f.flush()

with open(outputdir+"/refs_all.txt","w") as f:
    for ref in refs:
        f.write(ref+"\n")
        f.flush()


