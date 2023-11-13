## Machine-Learning-Concepts

### `One-Hot Encoding`
The Main purpose is to make easy for Machine Learning Algorithms beacause machine can understand only numerical values
One-hot Encoding is a technique used in Machine learning and data preprocessing to convert the `categorical` variables into `Numerical` variables (Binary numbers) 0s and 1s, where each category becomes a seperate binary feature. Each seperated features represents the presence(1) or obsence(0) of the particular category

### `Feature Engineering`
Feature Engineering is the process of `Creating new features` using the existing one or `Modifying the existing features` of the dataset to `improve the performance of a machine learning model` 


### `Outliers`
Before looking for `Outliers`, Let us first address the quetions like:
##### What are Outliers abd Why do we need to remove them ? 

`Outliers` are Nothing, But the points that are significatlty different from the majority of the data in a dataset. They can be usually high or low values that deviates from majority of the data distributionn. Now the second question is Why do we need to remove the outliers ? We need to remove the Outliers beacuase they can signifiantly influence the statistical measures like `Mean` and `Standard Deciation`. As we all know that the Mean is calculated by summingup all the values in the dataset and dividing by the total number of values. Outliers contribution is also there in calculation of the mean . `if there are high outliers, the mean tend to be higher than the  central tendency of the majority of the data`. Conversly, if there are lower outliers, the mean tends to be loewr than the central tendency of the majority of the data
