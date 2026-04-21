@echo off
if not exist pseudo mkdir pseudo
cd pseudo
curl -s -O -J -L https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-n-kjpaw_psl.0.3.0.UPF
curl -s -O -J -L https://pseudopotentials.quantum-espresso.org/upf_files/Mn.pbe-spn-kjpaw_psl.0.3.1.UPF
curl -s -O -J -L https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF
cd ..
