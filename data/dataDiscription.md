## MIT-BIH

+ 48 half-hour two-lead ECG recordings obtained from 48 patients
+  digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range
+ 41 labels for annotation for different meanings
+ one recording of the dataset contains 650000 sampling spots, and two leads which are 'MLII' and 'V5' respectively

## 心拍截取

+ 数据集的标记一共有44种类型，但是以心拍的R波标记位置的只有19种：
`'N', 'f', 'e', '/', 'j', 'n', 'B', 'L', 'R', 'S', 'A', 'J', 'a', 'V', 'E', 'r', 'F', 'Q', '?'`
+ 找到R波，前溯100个采样点，后取150个采样点，约0.7秒为一个心拍