@echo off
echo Starting Quantum Espresso batch execution...

for /d %%D in (*) do (
    echo Processing directory: %%D
    cd %%D
    
    echo [%%D] Downloading pseudopotentials...
    if not exist "pseudo" mkdir pseudo
    cd pseudo
    if not exist "Mg.pbe-n-kjpaw_psl.1.0.0.UPF" curl -s -O -J -L "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-n-kjpaw_psl.1.0.0.UPF"
    if not exist "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF" curl -s -O -J -L "https://pseudopotentials.quantum-espresso.org/upf_files/Mn.pbe-spn-kjpaw_psl.0.3.1.UPF"
    if not exist "O.pbe-n-kjpaw_psl.1.0.0.UPF" curl -s -O -J -L "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF"
    cd ..

    echo [%%D] Running pw.x (Quantum ESPRESSO)...
    :: Make sure 'pw.x' is in your system PATH, or provide the full absolute path below.
    :: If using MPI, prefix with: mpiexec -n 4 pw.x -in pw_relax.in > pw_relax.out
    
    pw.x -in pw_relax.in > pw_relax.out
    
    if %errorlevel% neq 0 (
        echo [ERROR] pw.x computation failed for %%D. Check if Quantum Espresso is installed and added to PATH.
    ) else (
        echo [SUCCESS] Completed %%D.
    )

    cd ..
    echo ----------------------------------------
)
echo All tasks executed.
pause
