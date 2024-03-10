3 Dimensional Convolutional Neural Network for Motor Imagery decoding. Multiclass (3), ALS patient data decoding. 
The data is provided by Penn State for 8 ALS patients over the course of 1-2 Months of monitoring spread across approx. 160 sessions.

Novelty: 
1) Application of deep learning structures to end user datasets. This contributes to benchmarks for models on ALS data.
2) Providing automatically adaptive models over time
3) 

Model To do list: 
# 1) 3D EfficientNet Frequency
| Subject | Accuracies (L/R/Re Respectively)              |<br>
|    1    | 0.6625 0.633333 0.970833                      |<br>
|    2    | [0.67054264 0.35271318 0.68217054]            |<br>
|    3    | 0.57563025 0.43277311 0.                      |<br>
|    4    | [0.59663866 0.54201681 0.87815126]            |<br>
|    5    | [0.62083333 0.6125     0.96666667]            |<br>
|    6    | [0.58333333 0.6375     0.8875    ]            |<br>
|    7    | [0.62666667 0.34222222 0.        ]            |<br>
|    8    | [0.30666667 0.60888889 0.        ]            |<br>

# 2) 2D EfficientNet - Raw Data
39 - [0.66222222 0.63111111 0.96      ]<br>
34 - [0.68444444 0.65333333 0.91555556]<br>
31 - [0.64583333 0.62083333 0.96666667]<br>
21 - [0.64583333 0.67083333 0.96666667]<br>
9 - [0.62605042 0.39915966 0.        ]<br>
5 - [0.61344538 0.31932773 0.        ]<br>
2 - [0.6744186  0.67054264 0.98062016]<br>
1 - [0.67083333 0.3125     0.        ]<br>
   
# 3) RNN - LSTM
# 4) LSTM + CNN (Simultaneous processing of frequency and raw data)
# 5) Using PLV as images for CNN (Start with All Freq, then just Mu and Beta)
# 6) Graph Neural Network - PLV with Raw EEG, and Power of each band as features per node. Graph Classification Task
| Subject | Accuracies (Train/Test)          |<br>
|    1    | 0.8375 0.65                      |<br>
|    2    | 0.9264 0.7132                    |<br>
|    3    | 0.7679 0.6708                    |<br>
|    4    | 0.7426 0.6458                    |<br>
|    5    | 0.7208 0.6375                    |<br>
|    6    | 0.6833 0.5792                    |<br>
|    7    | 0.9156 0.6089                    |<br>
|    8    | 0.8311 0.6133                    |<br>





