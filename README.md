**Function/Goal:**

This repo is for image analysis to measure particle size and shapes with the original intent being for TEM images. The image analysis script uses a classical computer vision approach to particle detention and measurements. As such, for some images even with input variables properly tweaked may not be compatible with quality measurements. The particle statistics script takes the output of the image analysis script or compiled manual measurements, converts pixel measurements to physical based on hard coded or in directory data and produce PDFs detailing the statistics of the peak shape and size (max/min Feret).





**Running the program- Particle Statistics:**

\-A directory that contains the excel measurement files should be created. See the reference excel file for what is being looked for by the program.

\-Ensure that no needed file has "compiled" in the name as the program is blind to these files to prevent it from reading its own output on subsequent runs.

\-Upon running the program input the filepath and select if you would like a statistical report and/or what time of measurement report to be outputted as PDFs.

\-The program will run and save PDF files to the inputed filepath.



**Running the program- TEM image analysis:**

\-In its current form this script is best to be run in something that allows for segregated block execution. This is because certain blocks are used to evaluate starting variable parameters the user can use to optimize measurements.

\-This script looks into the provided filepath and grabs the first or specified .jpg and gives some potential starting variables for measurements.

\-There are two execution blocks to evaluate images. One is for the initial image loaded and the other is for the entire directory. The goal is to create a directory with equivalent images and to optimize on one before analyzing the entire group at once.

\-Each block has a section for the variables with short explanations with what adjustments do to the measurements. Watershed particle separation is a Boolean for designating if the goal is to look for aggregates or single particles.



**Things to remember:**

**-------------------**

\-Having any of the needed files open will cause the program to crash. Closing the file and rerunning the program will fix that issue.

\-Additional runs will overwrite the previous graphs generated given.

\-Image analysis is through classical computer vision and may not produce great results for all images.

