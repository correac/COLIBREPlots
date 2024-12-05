#!/bin/bash

#python general_relations.py \
#-d /Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0/ /Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0p001/ \
#/Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0p01/ /Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0p1/ \
#-n C0 C0p001 C0p01 C0p1 \
#-c halo_0007.properties.0 halo_0007.properties.0 halo_0007.properties.0 halo_0007.properties.0 \
#-s colibre_0007.hdf5 colibre_0007.hdf5 colibre_0007.hdf5 colibre_0007.hdf5 \
#-o /Users/cc276407/Coding/COLIBREPlots/COLIBREPlots/outputs/

python general_relations.py \
-d /Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0/ /Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0p001/ \
/Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0p01/ /Users/cc276407/Simulation_data/cosma/COLIBRE/L25N376_C0p1/ \
-n C0 C0p001 C0p01 C0p1 \
-so SOAP_halo_properties_0007.hdf5 SOAP_halo_properties_0007.hdf5 SOAP_halo_properties_0007.hdf5 SOAP_halo_properties_0007.hdf5 \
-s colibre_0007.hdf5 colibre_0007.hdf5 colibre_0007.hdf5 colibre_0007.hdf5 \
-o /Users/cc276407/Coding/COLIBREPlots/COLIBREPlots/outputs/
