## Machine-Learning-Concepts

### `One-Hot Encoding`
The Main purpose is to make easy for Machine Learning Algorithms beacause machine can understand only numerical values
One-hot Encoding is a technique used in Machine learning and data preprocessing to convert the `categorical` variables into `Numerical` variables (Binary numbers) 0s and 1s, where each category becomes a seperate binary feature. Each seperated features represents the presence(1) or obsence(0) of the particular category

### `Feature Engineering`
Feature Engineering is the process of `Creating new features` using the existing one or `Modifying the existing features` of the dataset to `improve the performance of a machine learning model` 


### `Outliers`
Before looking for `Outliers`, Let us first address the quetions like:
##### What are Outliers abd Why do we need to remove them ? 

`Outliers` are Nothing, But the points that are significatlty different from the majority of the data in a dataset. They can be usually high or low values that deviates from majority of the data distributionn. Now the second question is Why do we need to remove the outliers ? We need to remove the Outliers beacuase they can significantly influence the statistical measures like `Mean` and `Standard Deciation`. As we all know that the Mean is calculated by summingup all the values in the dataset and dividing by the total number of values. Outliers contribution is also there in calculation of the mean . `if there are high outliers, the mean tend to be higher than the  central tendency of the majority of the data`. Conversly, if there are lower outliers, the mean tends to be loewr than the central tendency of the majority of the data. `Standard Deviation` measures the amount of dispersion in the dataset . Outliers can increase the standard deviation because they contribute to greater variability.leading to an inflated standard deviation

In a boxplot, outliers are often identified based on their position relative to the "whiskers" of the box. The whiskers extend from the box to indicate the range of the data, and points beyond a certain distance from the box are considered potential outliers.

Here's how outliers are typically identified in a boxplot:

Interquartile Range (IQR) Method:

The box in the boxplot represents the interquartile range (IQR), which is the range between the first quartile (Q1) and the third quartile (Q3).
The whiskers extend from the box to a certain distance, usually 1.5 times the IQR.
Any data points beyond the whiskers are considered potential outliers.
Outlier Calculation:

Lower Bound: 
�
1
−
1.5
×
�
�
�
Q1−1.5×IQR
Upper Bound: 
�
3
+
1.5
×
�
�
�
Q3+1.5×IQR
Identification:

Any data point below the lower bound or above the upper bound is considered a potential outlier and is often plotted individually as points.
Here's a breakdown of the components in a boxplot:

Box: Represents the interquartile range (IQR), which contains the middle 50% of the data. The line inside the box represents the median.

Whiskers: Extend from the box to the smallest and largest values within the "whisker length" from the quartiles. Points beyond the whiskers are considered potential outliers.

Outliers: Individual data points beyond the whiskers are plotted as individual points.
