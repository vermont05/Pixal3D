#!/bin/bash
# Runs run_pixal3d.sh with a RAM guard (stops the run if MemAvailable < 1.5 GB) and logs RAM and VRAM every 5 s.
# usage: guarded.sh <tag> <image> [resolution]
T=$1; L=~/src/pixal3d/out/$T.log; M=~/src/pixal3d/out/$T.mem; : > $M
~/src/pixal3d/run_pixal3d.sh "$2" ~/src/pixal3d/out/$T.glb "${3:-512}" > $L 2>&1 &
P=$!; minav=999999; maxv=0
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); v=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used)/1048576 ))
  [ $a -lt $minav ] && minav=$a; [ $v -gt $maxv ] && maxv=$v; echo "$(date +%T) avail=$a vram=$v" >> $M
  if [ $a -lt 1500 ]; then echo "MEMABORT avail=$a" >> $L; kill $P; sleep 5; kill -9 $P 2>/dev/null; break; fi
  sleep 5
done
wait $P; rc=$?
echo "RESULT rc=$rc min_avail_MB=$minav peak_vram_MB=$maxv" | tee -a $L
