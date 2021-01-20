# DomainDecomposition

Download deal.II (available at https://dealii.org/download.html) and follow deal.II documentation for installation.
Where the other deal.II example programs are saved (dealii/examples), create a copy of the Step-6 (CopiedFolder). This is where you should keep this modified Step6 file and its output files. 
In CopiedFolder, replace the step6.cc file with the most-updated file from my DomainDecomposition GitHub repository available at https://github.com/christina-rigsby/DomainDecomposition   

Using your IDE of choice (I am using CLion, available for download at https://www.jetbrains.com/clion/download/#section=windows), open this new step6.cc file. From here, you can run the program, edit it, or do both.

To run the program, open a terminal and navigate to step6.cc's location (should be in dealii/examples/CopiedFolder).
Make and run this file by typing: cmake . && make run

Now that the file is made, you only need to type: make run
to run your program with any modifications you have made in your IDE.

The program output will appear in CopiedFolder as well and can be visualized with VisIT, which can be run by simply typing: visit
in the terminal.
