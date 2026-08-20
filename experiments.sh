#!/bin/bash
# Runs every eval_f.py / eval_graph.py evaluation in series.
# Set DATA and CODE below to match your data location and checkout.

set -euo pipefail

DATA=/home/kocurvik/data/rd
CODE=/home/kocurvik/code/rdnet

FEAT=pairs-features_superpoint_noresize_2048-LG
FEAT_EQ=pairs-features_superpoint_noresize_2048-LG_eq

NW=64                       # -nw, worker processes
SARGS=$(seq 1 3)            # -s values: 1=synthA, 2=synthB, 3=synthC
CHARS=('' 'A' 'B' 'C')      # -s value -> synth character, as in eval_f.py

ETH3D=$DATA/ETH3D/multiview_undistorted
ROT=$DATA/rotunda_new
CAT=$DATA/cathedral
EUROC=$DATA/Euroc
PRAGUE=$DATA/pragueparks

cd "$CODE"
export PYTHONPATH=$CODE

# ETH3D (synthetic distortion; feature files carry a "-final" infix)
for sarg in $SARGS; do
    c=${CHARS[$sarg]}
    for x in $(ls "$ETH3D"); do
        echo "=== ETH3D $x synth$c ==="
        python eval_f.py --net    -s "$sarg" -nw $NW "$ETH3D/$x/synth$c-uneq-final-$FEAT" "$ETH3D/$x"
        python eval_f.py --net -e -s "$sarg" -nw $NW "$ETH3D/$x/synth$c-eq-final-$FEAT"   "$ETH3D/$x"
    done
done

# rotunda
echo "=== rotunda ==="
python eval_f.py     --net -nw $NW -e "$ROT/$FEAT_EQ" "$ROT"
python eval_f.py     --net -nw $NW    "$ROT/$FEAT"    "$ROT"
python eval_graph.py       -nw $NW -e "$ROT/$FEAT_EQ" "$ROT"
python eval_graph.py       -nw $NW    "$ROT/$FEAT"    "$ROT"

# cathedral
echo "=== cathedral ==="
python eval_f.py     --net -nw $NW -e "$CAT/$FEAT_EQ" "$CAT"
python eval_f.py     --net -nw $NW    "$CAT/$FEAT"    "$CAT"
python eval_graph.py       -nw $NW -e "$CAT/$FEAT_EQ" "$CAT"
python eval_graph.py       -nw $NW    "$CAT/$FEAT"    "$CAT"

# euroc
for x in $(ls "$EUROC"); do
    echo "=== euroc $x ==="
    python eval_f.py --net -nw $NW -e "$EUROC/$x/features" "$EUROC/$x"
done

# pragueparks (synthetic distortion; no "-final" infix here)
for x in $(ls "$PRAGUE"); do
    for sarg in $SARGS; do
        c=${CHARS[$sarg]}
        echo "=== pragueparks $x synth$c ==="
        python eval_f.py --net    -s "$sarg" -nw $NW "$PRAGUE/$x/synth$c-uneq-$FEAT" "$PRAGUE/$x"
        python eval_f.py --net -e -s "$sarg" -nw $NW "$PRAGUE/$x/synth$c-eq-$FEAT"   "$PRAGUE/$x"
    done
done