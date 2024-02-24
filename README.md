3 Dimensional Convolutional Neural Network for Motor Imagery decoding. Multiclass (3), ALS patient data decoding. 
The data is provided by Penn State for 8 ALS patients over the course of 1-2 Months of monitoring spread across approx. 160 sessions.

Novelty: 
1) Application of deep learning structures to end user datasets. This contributes to benchmarks for models on ALS data.
2) Providing automatically adaptive models over time
3) 

Model To do list: 
# 1) 3D EfficientNet Frequency
| Subject | Accuracies (L/R/Re Respectively)              |
|    1    | 0.6625 0.633333 0.970833                      |
|    2    | [0.67054264 0.35271318 0.68217054]            |
|    3    | 0.57563025 0.43277311 0.                      |
|    4    | [0.59663866 0.54201681 0.87815126]            |
|    5    | [0.62083333 0.6125     0.96666667]            |
|    6    | [0.58333333 0.6375     0.8875    ]            |
|    7    | [0.62666667 0.34222222 0.        ]            |
|    8    | [0.30666667 0.60888889 0.        ]            |

# 2) 2D EfficientNet - Raw Data
39 - [0.66222222 0.63111111 0.96      ]
34 - [0.68444444 0.65333333 0.91555556]
31 - [0.64583333 0.62083333 0.96666667]
21 - [0.64583333 0.67083333 0.96666667]
9 - [0.62605042 0.39915966 0.        ]
5 - [0.61344538 0.31932773 0.        ]
2 - [0.6744186  0.67054264 0.98062016]
1 - [0.67083333 0.3125     0.        ]
   
# 3) RNN - LSTM
# 4) LSTM + CNN (Simultaneous processing of frequency and raw data)
# 5) Using PLV as images for CNN and LSTM?






