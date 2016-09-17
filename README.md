# Person Re-identification in Appearance Impaired Scenarios
###### Mengran Gou, Xikang Zhang, Angels Rates-Borras, Sadjad Asghari-Esfeden, Octavia Camps, Mario Sznaier
###### Northeastern University, Boston, USA
### Abstract
Person re-identification is critical in surveillance applications. Current approaches rely on appearance-based features extracted from a single or multiple shots of the target and candidate matches. These approaches are at a disadvantage when trying to distin- guish between candidates dressed in similar colors or when targets change their clothing. In this paper we propose a dynamics-based feature to overcome this limitation. The main idea is to capture soft biometrics from gait and motion patterns by gathering dense short trajectories (tracklets) which are Fisher vector encoded. To illustrate the merits of the proposed features we introduce three new “appearance-impaired” datasets. Our experiments demonstrate the benefits of incorporating dynamics-based information into re-identification algorithms.

![Flow chart](https://github.com/NEU-Gou/dynamicreid/blob/master/flowchart.png "DynFV")
### How to run
1. Clone this git;
2. Down the data from [here](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/code/PRID_allbk_35.zip);
3. Create folders "Dataset" and move "PRID_Images_Tracklets_l15_allbk_35.mat" into it;
4. Create folder "Feature" and move "PRID_Partition_Random_allbk_35.mat" and "PRID_semEdge_allbk_35.mat" into it;
5. Download and install [vl_feat](http://www.vlfeat.org);
6. Run example file "test_Ranking_dynamic.m"

NOTE: Due to the randomness in GMM part, one cannot get exactly the same result as we reported in the paper. 

### BibTex
```
@inproceedings{gou2016person,
title={Person Re-identification in Appearance Impaired Scenarios},
author={Mengran Gou, Xikang Zhang, Angels Rates-Borras, Sadjad Asghari-Esfeden, Octavia Camps, Mario Sznaier},
booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
year={2016}
}
```