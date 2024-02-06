#!/bin/bash

set -x

folder=""
stack=$(ls $folder/. | grep -v py)
path_file=""
CMD="main_dust.py"

for i in $stack; do
	$CMD -fd $folder -sd $i -pf $path_file -wvl N -pix N -interval N -np N -zeta N -st N -lim N -msk N -par1 N -par2 N -dimx N -dimy N -sample "" -ray N -area N -cext v
done


exit 0
