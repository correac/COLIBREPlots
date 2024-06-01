COLIBREPlots
=========

A python package that makes stellar abundance plots from the COLIBRE simulations.

Requirements
----------------

The colibre-plots package requires:

+ `python3.9` or above
+ see requirements.txt

Usage
---------------

To run the script in the _single-run_ mode use
```bash
 python3 colibre_plots.py -d run_directory \
                          -s snapshot_name \
                          -c catalogue_name \
                          -n name_of_the_run \
                          -o path_to_output_directory 
```

To run the script in the _comparison_ mode use
```bash
 python3 colibre_plots.py -d directory_of_run1 directory_of_run2 \
                          -s snapshot_name_run1 snapshot_name_run2 \
                          -c catalogue_name_run1 catalogue_name_run2 \
                          -n name_of_the_run1 name_of_the_run2 \
                           -o path_to_output_directory
```



