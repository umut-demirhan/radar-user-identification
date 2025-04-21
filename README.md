# Enabling ISAC in Real World: Beam-Based User Identification with Machine Learning
This is a python code package related to the following article:
Umut Demirhan, and Ahmed Alkhateeb, [Enabling ISAC in Real World: Beam-Based User Identification with Machine Learning](https://arxiv.org/abs/2411.06578)," IEEE Wireless Communications Letters, 2025.

# Instructions to Reproduce the Results 
The scripts for generating the results of the solutions in the paper. This script adopts Scenario 35 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**
1. Download [Scenario 35 of the DeepSense 6G dataset](https://www.deepsense6g.net/scenario-35/).
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory 
   (If needed, the dataset directory can be changed at line 24 of main.py)
5. Run main.py file.

If you have any questions regarding the code and used dataset, please write to DeepSense 6G dataset forum https://deepsense6g.net/forum/ or contact [Umut Demirhan](mailto:udemirhan@asu.edu?subject=[GitHub]%20Radar%20User%20Identification%20implementation).

# Abstract of the Article
Leveraging perception from radar data can assist multiple communication tasks, especially in highly-mobile and large-scale MIMO systems. One particular challenge, however, is how to distinguish the communication user (object) from the other mobile objects in the sensing scene. This paper formulates this \textit{user identification} problem and develops two solutions, a baseline model-based solution that maps the angle of the object from the radar scene to communication beams and a scalable deep learning solution that is agnostic to the number of candidate objects. Using the DeepSense 6G dataset, which has real-world measurements, the developed deep learning approach achieves more than $89\%$ communication user identification accuracy on the test set, highlighting a promising path for enabling integrated radar-communication  applications in the real world. 

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> U. Demirhan and A. Alkhateeb, "[Enabling ISAC in Real World: Beam-Based User Identification with Machine Learning](https://arxiv.org/abs/2411.06578)," IEEE Wireless Communications Letters, 2025.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, J. Morais, U. Demirhan, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” IEEE Communications Magazine, vol. 61, no. 9, pp. 122–128, 2023.
