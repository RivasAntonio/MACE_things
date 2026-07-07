#!/bin/bash
rm -rf all.xyz  
for file in xml/*_vasprun.xml ; do
 echo $file
 python collect.py --xml $file --out all.xyz --every 5 --verbose --append
done
rm -rf splits/*
python split_ase.py
mv *.xyz splits/.
exit 0
