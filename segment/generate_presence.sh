#!/bin/sh
rm presence.csv
echo '>>> make'
gcc segment_map.c -O2 -lm -o segment_map.bin
echo '>>> interpolate'
for i in echo /home/ngaude/workspace/data/arzephir_italy_place_segment_2014-05-*; do echo 'interpolate : '$i ; ./segment_map.bin $i; done
echo '>>> image'
rm /home/ngaude/workspace/data/image/*.png
python presence_imshow.py
echo '>>> movie'
mencoder mf:///home/ngaude/workspace/data/image/*.png -mf fps=12:type=png -o presence.avi -ovc xvid -xvidencopts bitrate=4000 -oac copy
