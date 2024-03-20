#!/bin/python

import sys


pred=sys.argv[1]
ref=sys.argv[2]
con=sys.argv[3]


def read(fil):
  lines=[]
  with open(fil,"r") as f:
       for line in f:
          lines.append(line.strip())
  return lines

predlines=read(pred)
reflines=read(ref)
conlines=read(con)

for i in range(len(predlines)):
    prediction=predlines[i]
    reference=reflines[i]
    context=conlines[i]
    print("Context:", context)
    print("Prediction:", prediction)
    print("Reference:", reference)
    input("")


