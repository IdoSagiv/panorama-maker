# Panorama Maker
This program converts mp4 videos into a panorama.<br/>
<p align="center">
    <img src="images/boat_panorama.gif">
</p>
* the original video can be found [here](https://github.com/IdoSagiv/panorama-maker/blob/main/videos/boat.mp4)
## Technologies And Tools
This program is written in Python and using the libraries: 'numpy', 'mayplotlib', 'scipy' and 'imageio'.<br/>
The program was developed on Linux, and tested both over Linux and Windows machines.
## Overview
The panorama making process is:
1. Dividing the input video into frames
2. Finding feature points to every frame
3. Finding matching feature points to every two consecutive frames
4. Aligning the frames to a common coordinate system using the [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) algorithm
5. Stitching the aligned frames by their matching points

## Key Features
### Finding feature points
Features are found using the [Harris Corner Detector](https://en.wikipedia.org/wiki/Harris_Corner_Detector) algorithm, and a [MOPS-like](https://www.cs.cornell.edu/courses/cs6670/2011sp/projects/p1/webpages/6/webpage.html) descriptor is extracted from each feature point.<br/>
<p align="center">
    <img src="images/oxford1_feature_points.png" width="450">
    <img src="images/oxford2_feature_points.png" width="450">
</p>

### Finding matching points between two images
Using a comparison between the [MOPS-like](https://www.cs.cornell.edu/courses/cs6670/2011sp/projects/p1/webpages/6/webpage.html) descriptors of the feature points of each image.<br/>
![oxford_matching_points](https://github.com/IdoSagiv/panorama-maker/blob/main/images/oxford_matching_points.png?raw=true)<br/>
### Different perspectives
By taking strips from different locations in the frames and stitch them using the matching feature points, different perspectives of the scene can be made.<br/>
![first_perspective](https://github.com/IdoSagiv/panorama-maker/blob/main/perspective_panoramic_frames/boat/panorama01.png?raw=true)<br/>
![second_perspective](https://github.com/IdoSagiv/panorama-maker/blob/main/perspective_panoramic_frames/boat/panorama09.png?raw=true)<br/>
