Ensemble Feature Selection Framework for Network Intrusion Detection Systems
This repository provides the implementation of our ensemble-based feature selection framework, leveraging Explainable AI (XAI) methods to enhance the performance of network intrusion detection systems (IDS). 
The framework combines state-of-the-art XAI methods to rank feature importance and employs a frequency-based aggregation mechanism to identify critical features. 


Highlights
Explainable AI Integration: Combines SHAP, LOCO, PFI, and DALEX for feature importance rankings.
Frequency-Based Aggregation: Identifies and prioritizes features consistently ranked high across methods.
Extensive Evaluation: Tested on the CICIDS-2017 dataset using multiple classifiers, including Random Forest, Logistic Regression, KNN, and AdaBoost.
Open Source Contribution: Source code available for the research community to advance and adapt.

**Our Framework**
![XAI Features importance selection](https://github.com/user-attachments/assets/89f075cc-138c-4c5a-9433-49b842b35e77)


To Download the CICIDS-2017: dataset https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17.

Our Results
The frequency-based ensemble framework significantly enhances model performance while reducing computational costs. Below are key performance metrics:

Classifier	Accuracy	Precision	F1 Score	Computational Efficiency
Random Forest	99.83%	99.83%	99.83%	Optimized
Logistic Regression	91.85%	87.70%	89.45%	High
For detailed results, see the evaluation report.



Ensemble Feature Selection for Network Intrusion Detection Systems Using Explainable AI: A Frequency-Based Approach
Ismail Bibers & Mustafa Abdallah


Contributing
We welcome contributions from the community!



Contact
For questions, comments, or collaborations, reach out to:

Ismail Bibers: ibibers@purdue.edu
Mustafa Abdallah: abdalla0@purdue.edu
