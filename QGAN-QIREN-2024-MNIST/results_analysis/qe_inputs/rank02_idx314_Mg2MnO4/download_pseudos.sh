#!/bin/bash
mkdir -p pseudo
cd pseudo
wget -nc https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-n-kjpaw_psl.1.0.0.UPF
wget -nc https://pseudopotentials.quantum-espresso.org/upf_files/Mn.pbe-spn-kjpaw_psl.0.3.1.UPF
wget -nc https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF
cd ..
