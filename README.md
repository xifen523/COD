### <p align="center"> Towards Consistent Object Detection via LiDAR-Camera Synergy 
<br>
<div align="center">
  Kai&nbsp;Luo*</a> <b>&middot;</b>
  Hao&nbsp;Wu*</a> <b>&middot;</b>
    <a href="https://www.csust.edu.cn/jtysgcxy/info/1130/17551.htm" target="_blank">Kefu&nbsp;Yi</a> <b>&middot;</b>
  Kailun&nbsp;Yang</a> <b>&middot;</b>
  Wei&nbsp;Hao</a> &middot;</b>
  Rongdong&nbsp;Hu</a> &middot;</b>

  
  <br> <br>
  <a href="https://arxiv.org/pdf/2405.01258.pdf" target="_blank">Paper</a>
</div>

<br>
<p align="center">Code will be released soon. </p>
<br>

<div align=center><img src="imgs/overall.png" /></div>

### Update
- 2024.04.30 Init repository.
- 2024.07.11 COD [(PDF)](https://arxiv.org/pdf/2405.01258.pdf) is accepted to SMC2024

### Abstract

As human-machine interaction continues to evolve, the capacity for environmental perception is becoming increasingly crucial. Integrating the two most common types of sensory data, images, and point clouds, can enhance detection accuracy. Currently, there is no existing model capable of detecting an object's position in both point clouds and images while also determining their corresponding relationship. This information is invaluable for human-machine interactions, offering new possibilities for their enhancement. In light of this, this paper introduces an end-to-end Consistency Object Detection (COD) algorithm framework that requires only a single forward inference to simultaneously obtain an object's position in both point clouds and images, and establish their correlation. Furthermore, to assess the accuracy of the object correlation between point clouds and images, this paper proposes a new evaluation metric, Consistency Precision (CP). To verify the effectiveness of the proposed framework, an extensive set of experiments has been conducted on the KITTI and DAIR-V2X datasets. The study also explored how the proposed consistency detection method performs on images when the calibration parameters between images and point clouds are disturbed, compared to existing post-processing methods. The experimental results demonstrate that the proposed method exhibits excellent detection performance and robustness, achieving end-to-end consistency detection. The source code will be made publicly available at [https://github.com/xifen523/COD](https://github.com/xifen523/COD).

