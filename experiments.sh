# change /path/to/stored_matches/ to your path
# set -nw to your number of cores, setting -nw 1 runs the code without parallelization (for debugging)

for sarg in $( seq 2 2); do
    for x in $( ls /path/to/stored_matches/ETH3D/multiview_undistorted); do
        echo $x
        python eval_f.py --net -s $sarg -nw 64 /path/to/stored_matches/ETH3D/multiview_undistorted/$x/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/ETH3D/multiview_undistorted/$x
        python eval_f.py --net -e -s $sarg -nw 64 /path/to/stored_matches/ETH3D/multiview_undistorted/$x/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/ETH3D/multiview_undistorted/$x
    done
done

# rotunda
python eval_f.py --net -nw 64 -e /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/rotunda_new
python eval_f.py --net -nw 64 /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/rotunda_new

# rotunda graph
python eval_graph.py -nw 64 -e /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/rotunda_new
python eval_graph.py -nw 64 /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/rotunda_new

# cathedral
python eval_f.py --net -nw 64 -e /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/cathedral
python eval_f.py --net -nw 64 /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/cathedral

# cathedral
python eval_graph.py -nw 64 -e /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/cathedral
python eval_graph.py -nw 64 /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/cathedral

# euroc
for x in $( ls /path/to/stored_matches/Euroc); do
    echo $x
    python eval_f.py --net -nw 64 -e /path/to/stored_matches/euroc/$x/features /path/to/stored_matches/euroc/$x
done



