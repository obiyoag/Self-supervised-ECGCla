## MIT-BIH

+ 48 half-hour two-lead ECG recordings obtained from 48 patients
+  digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range
+ 41 labels for annotation for different meanings
+ one recording of the dataset contains 650000 sampling spots, and two leads which are 'MLII' and 'V5' respectively

## 心拍截取

数据集的标记一共有44种类型，但是以心拍的R波标记位置的只有19种：
`'N', 'f', 'e', '/', 'j', 'n', 'B', 'L', 'R', 'S', 'A', 'J', 'a', 'V', 'E', 'r', 'F', 'Q', '?'`  
按照AAMI标准分成5大类：  
+ N:0
    + N: normal beat 正常心搏
    + L: left bundle branch block beat 左束支传导阻滞心搏
    + R: right bundle branch block beat 右束支传导阻滞心搏 
+ SVEB:1
    + S: premature or ectopic supraventricular escape beat 早搏或室上性异位心搏
    + A: atrial premature beat 房性早搏
    + J: nodal premature beat 交界性早搏
    + a: abberated atrial premature beat 异常房性早搏
    + e: atrial escape beat 房性逸搏
    + j: nodal escape beat 交界性逸搏
+ VEB:2
    + V: premature ventricular contraction 室性早搏
    + E: ventricular escape beat 室性逸搏
+ F:3
    + F: fusion of ventricular and normal beat 心室融合心搏
+ Q:4
    + \: paced beat 起搏心拍
    + f: fusion of paced and normal beat 起搏融合心跳
    + Q: unclassified beat 未分类心拍


找到R波，前溯100个采样点，后取150个采样点，约0.7秒为一个心拍

因为有4段recording用了起搏器， 按照AAMI标准排除在外， 分别是102,104,107和217  

分类过后共有心拍100578段：
+ N:89841      label:0
+ S:2927       label:1
+ V:7008       label:2
+ F:802        label:3

因为Q只有15段，就没有分出来  

现在进一步做的数据集的划分。  
分成三个数据集：
1. 用于心电信号reconstruction的训练集:MaskSet  1800:1800:1800:600   将每一段信号中的100-150置0  
2. 用于分类器训练的训练集                       600:600:600:160
3. 用于测试的测试集                            600:527:600:42

这样划分训练效果有没有自监督任务差别不大，可能是因为fine_tuning用的训练集过大  
下面考虑fine_tuning时只用数据集2，划分以不同的比例划分训练和测试集  
+ 当训练和测试集比例为0.5:0.5时，自监督acc:0.99, 无自监督acc:0.90