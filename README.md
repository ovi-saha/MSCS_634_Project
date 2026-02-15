# MSCS_634_Project

**Group Members:**  
Avijit Saha  
Pranoj Thapa  
Sandip KC  
Bharath Singareddy  

**Advanced Big Data and Data Mining (MSCS-634-M20)**  

Dr. Satish Penmatsa  
February 15, 2026


## Project Overview

This project explores passenger travel data using advanced data mining and machine learning techniques. The goal is to analyze patterns, classify travel behaviors, segment data, and identify meaningful associations. Deliverable 4 consolidates insights from previous deliverables (1–3) and presents a comprehensive analysis of the dataset, methodologies used, results obtained, and actionable recommendations for transportation planning.

The project addresses three major areas:

1. **Classification Analysis** – Predicting travel-related categories such as Low, Medium, and High.
2. **Clustering Analysis** – Identifying natural groupings in the data using unsupervised learning.
3. **Association Rule Mining** – Discovering relationships between travel modes, passenger statistics, and years.

## Dataset

The dataset contains passenger travel data spanning multiple years and transportation modes. Key attributes include:

- **Year**: Year in which the data was recorded.
- **Mode**: Mode of transportation, including Air, Road, Rail, etc.
- **Statistic**: Travel-related statistics such as Number of Passengers, Injured Persons, Fatalities, and Licensed Drivers.
- **Value**: Numeric representation of the statistic.
- **Additional features** include variables relevant to safety and passenger demographics.

Before analysis, the dataset was preprocessed by handling missing values, encoding categorical variables into numeric form, and scaling features for clustering and classification.

## Data Preprocessing

Data preprocessing was performed in multiple steps to prepare the dataset for analysis:

1. Missing or invalid values were removed.
2. Categorical variables, such as transportation **Mode**, were encoded into numerical representations for machine learning algorithms.
3. Feature selection was applied to focus on the most important variables, including **Year, Mode, and Statistic**.
4. Numerical features were scaled to ensure fair distance calculations for K-Means and K-Nearest Neighbors (KNN) models.
5. The cleaned dataset was split into training and testing sets for supervised learning tasks.

These steps ensured that the data was clean, consistent, and suitable for modeling.

## Deliverable 1: Exploratory Data Analysis (EDA)

Exploratory data analysis (EDA) was conducted to understand data distribution, trends, and outliers. Key steps included:

- **Descriptive Statistics**: Mean, median, and standard deviation were calculated for numerical features.
- **Visualizations**: Histograms, scatter plots, and line charts were used to visualize yearly trends and distribution by transportation mode.
- **Insights**: Observations revealed fluctuations in passenger counts over the years, differences between transportation modes, and patterns in safety-related statistics.

EDA provided the foundation for subsequent classification, clustering, and association analysis by highlighting relevant patterns and guiding feature selection.

## Deliverable 2: Clustering Analysis

K-Means clustering was applied to identify natural groupings in the passenger travel data. The process included:

1. **Elbow Method**: Tested cluster numbers ranging from 1 to 10 using the inertia metric to identify the optimal number of clusters.
2. **K-Means Implementation**: Using `n_clusters=3`, the algorithm segmented data into three clusters.
3. **Visualization**: Scatter plots illustrated cluster separation based on **Year** and **Value**, helping identify groupings of travel statistics.

**Observations:**

- The clusters revealed groups of similar travel patterns over time.
- Some clusters contained high-value statistics corresponding to higher travel volumes or safety concerns, while others captured lower-value statistics.
- Clustering allowed for segmentation that can be used in planning and risk assessment.

**Interpretation:**

- K-Means clustering effectively uncovered hidden structures in the dataset.
- The insights can help transportation authorities target interventions, allocate resources, and prioritize safety measures.

## Deliverable 3: Classification Analysis

Classification models were developed to categorize passenger travel statistics into Low, Medium, and High classes. Three models were applied:

### 1. Decision Tree Classifier

**Objective:** Classify travel statistics and evaluate model performance.  

**Process:** Cleaned data, selected features, converted categorical variables, split into training/testing sets, trained the Decision Tree, and performed hyperparameter tuning using GridSearchCV.  

**Observations:**

- High and Low categories were classified accurately, Medium values showed more misclassification.
- Hyperparameter tuning improved model accuracy and reduced overfitting.
- Important features included Year and Mode.

**Interpretation:** Decision Trees are effective for understanding patterns in passenger data and provide reliable predictions for decision-making.

### 2. K-Nearest Neighbors (KNN)

**Objective:** Classify data based on similarity.  

**Process:** Selected features, scaled data, split into training/testing sets, optimized K, trained KNN model, and evaluated performance.  

**Observations:**

- KNN performed well with appropriate K value.
- Sensitive to outliers and data scaling.
- Medium category showed higher misclassification.

**Interpretation:** KNN is useful for pattern recognition and comparison but less suitable for very large datasets.

### 3. Naive Bayes Classifier

**Objective:** Use probability-based classification.  

**Process:** Cleaned data, selected features, split data, trained Naive Bayes, calculated probabilities, predicted classes, and evaluated results.  

**Observations:**

- Fast training and prediction.
- Works well for independent features and large datasets.
- Slightly lower accuracy compared to Decision Tree.

**Interpretation:** Naive Bayes is suitable as a baseline model and for quick initial analysis.

### Hyperparameter Tuning

Decision Tree hyperparameters such as `max_depth` and `min_samples_split` were tuned using GridSearchCV to improve model accuracy. Tuning resulted in:

- Reduced overfitting
- Balanced tree structure
- Improved prediction reliability

**Screenshot Suggestion:** Include a screenshot of the Decision Tree model visualization or ROC curve here.

## Deliverable 3: Association Rule Mining

Association rules were generated to discover relationships in the dataset:

- **Method:** Apriori algorithm with minimum support and confidence thresholds.
- **Findings:** Relationships were observed between **Mode**, **Year**, and passenger statistics.
- **Insights:** Such patterns can inform policy-making, transportation planning, and resource allocation.

**Screenshot Suggestion:** Include a table or chart showing the top association rules discovered.

## Deliverable 4: Consolidated Insights

Deliverable 4 consolidates all findings and provides actionable recommendations:

- Classification models identified high-risk segments and predicted travel behaviors.
- Clustering revealed natural groupings of similar travel statistics over time.
- Association rules uncovered significant relationships that can guide strategic decisions.
- Visualizations helped communicate insights effectively.

**Recommendations:**

- Use classification outputs to monitor and predict high-risk segments.
- Leverage cluster analysis for resource allocation and scheduling.
- Apply association rules for safety and policy improvement.
- Expand dataset features in future analyses for enhanced prediction and clustering.

## Project Visualizations

- Decision Tree plots, KNN confusion matrices, ROC curves
- K-Means cluster scatter plots
- Association rule tables or network graphs

## Submission Components

- **Jupyter Notebook (.ipynb):** Contains all preprocessing, modeling, and visualization code with comments.
- **Comprehensive Report (PDF):** Includes introduction, methodology, results, insights, and references.
- **Video Presentation :** Explains project journey, decisions, challenges, and insights.
- **README.md:** Summarizes the entire project, including dataset, steps, findings, and recommendations.

## References

- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Tan, P.-N., Steinbach, M., & Kumar, V. (2005). *Introduction to Data Mining*. Pearson.

## Presentation Link
https://go.screenpal.com/watch/cOno2En0qU1
