## Machine-Learning-Concepts

## `Linear Regression`
Linear Regression is a fundamental supervised machine learning model used for `predicting the continuous ( Numerical ) values of dependant variable` based on understanding the `best possible relationship ( hidden patterns ) between the one or more independent variables and dependent variable`
so, What I mean is that the model/computer/ machine 
   1. first look at the labeled data for the best possible relationship between the independent variables and dependent variable
   2. Predicting continuous values based on its finding.

### `Problem Statement`: 
Let us consider the simple problem of house price prediction to understand the linear regression easily. 
Imagine we are trying to predict house prices based on several factors:

   *  Size of House
   
   *  Number of Bedrooms 
   
   *  Distance to the city center
   
   *  Age of House 

 Consider the following table consisting of labeled data from which the machines would like to learn ( Understand ) how the features affect output `House Size (sq ft)`


| House Size (sq ft) | Number of Bedrooms | Age of House (years) | Distance to City Center (miles) | House Price ($1000s) |
|--------------------|--------------------|----------------------|---------------------------------|----------------------|
| 2000               | 3                  | 10                   | 5                               | 300                  |
| 1500               | 2                  | 5                    | 10                              | 200                  |
| 2500               | 4                  | 20                   | 2                               | 400                  |
| 1800               | 3                  | 15                   | 8                               | 250                  |
| 2200               | 4                  | 8                    | 6                               | 350                  |


#### `Now, What is the price of a house that’s 3000 sq ft, has 4 bedrooms, is 10 years old, and is 5 miles from the city center? `

| House Size (sq ft) | Number of Bedrooms | Age of House (years) | Distance to City Center (miles) | House Price ($1000s) |
|--------------------|--------------------|----------------------|---------------------------------|----------------------|
| 3000               | 4                  | 10                   | 5                               | What is Price ?      |
   
## `Gradient Descent Algorithm`
- It is an optimization algorithm used in not only Machine learning but also in Deep learning to `minimize the error ( cost function or loss function )` associated with the  model. it is a technique used for finding the best model's parameters Weights and Biases to reduce ( minimize ) the cost function or loss function.
-  ### How does this Work, I mean What is the actual work behind the Gradient Descent Algorithm?
   Let us train the model with the Linear function f(x[i]) for the Linear Regression problem and we have the cost function $J(w,b)$ which needs to be minimized by adjusting (w, b)

 $$f(x^{(i)}) = wx^{(i)} + b$$
 

 As we know that cost function $J(w,b)$ is a measure of the error between the predicted value $f(x^{i)})$ and actual value $y^{(i}) over all the training values $x^{(i)}, y^{(i)}$

 $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f(x^{(i)}) - y^{(i)})^2$$
 
Now, Our main objective is to find the best values of (w,b) such that the value of the cost function $J(w,b)$ will be minimum. In this Algorithm what we are doing is, we are changing the values of w, b until we get the minimum value of the $J(w,b)$

updating the values of the parameter w, b for each iteration using the mathematical equation :

$$ w = w - \alpha\frac{\partial J(w,b)}{\partial w}$$

$$ b = b - \alpha\frac{\partial J(w,b)}{\partial b}$$



Here, the Gradient term represents the `Magnitude` and `Direction`. Basically, `it tells us How to change the parameters w, b to reduce the cost`. Whereas $\alpha$   is `Learning Rate` Which is the Hyperparameter that determines the size of the step that we need to take during each iteration to get minima of the cost function
- Larger learning rate: A larger learning rate can cause the algorithm to `Converge more quickly`, but it `Oversoot` the minimum value of the cost function
- Smaller learning rate: on the other hand, a smaller learning rate may `Converge more slowly` but `potentially more precision` 


## `One-Hot Encoding`

The main purpose is to make it easy for Machine Learning Algorithms because machines can understand only numerical values
One-hot Encoding is a technique used in Machine learning and data preprocessing to convert the `categorical` variables into `Numerical` variables (Binary numbers) 0s and 1s, where each category becomes a separate binary feature. Each separated features represents the presence(1) or absence(0) of the particular category

## `Feature Engineering`
Feature Engineering is the process of `Creating new features` using the existing ones or `Modifying the existing features` of the dataset to `improve the performance of a machine learning model` 


## `Outliers`
Before looking for `Outliers, Let us first address the questions:
#### What are Outliers and Why do we need to remove them? 

`Outliers` are Nothing, But the points that are significantly different from the majority of the data in a dataset. They can be usually high or low values that deviate from the majority of the data distributions. Now the second question is Why do we need to remove the outliers? We need to remove the Outliers because they can significantly influence the statistical measures like `Mean` and `Standard Deviation`. As we all know the Mean is calculated by summing up all the values in the dataset and dividing by the total number of values. Outlier's contribution is also there in the calculation of the mean . `if there are high outliers, the mean tends to be higher than the  central tendency of the majority of the data`. Conversely, if there are lower outliers, the mean tends to be lower than the central tendency of the majority of the data. `Standard Deviation` measures the amount of dispersion in the dataset. Outliers can increase the standard deviation because they contribute to greater variability. leading to an inflated standard deviation

#### How Can we Identify the Potential Outliers from the box plots?

In a boxplot, outliers are often identified based on their position relative to the "whiskers" of the box. The whiskers extend from the box to indicate the range of the data, and points beyond a certain distance from the box are considered potential outliers.

Here's how outliers are typically identified in a boxplot:

#### `Interquartile Range (IQR) Method`:

The box in the boxplot represents the interquartile range (IQR), which is the range between the first quartile (Q1) and the third quartile (Q3).
The whiskers extend from the box to a certain distance, usually 1.5 times the IQR.
Any data points beyond the whiskers are considered potential outliers.

Outlier Calculation:

Lower Bound = Q1−1.5×IQR

Upper Bound = Q3+1.5×IQR

Identification:

Any data point below the lower bound or above the upper bound is considered a potential outlier and is often plotted individually as points.
Here's a breakdown of the components in a boxplot:

`Box`: Represents the interquartile range (IQR), which contains the middle 50% of the data. The line inside the box represents the median.

`Whiskers`: Extend from the box to the smallest and largest values within the "whisker length" from the quartiles. Points beyond the whiskers are considered potential outliers.

`Outliers`: Individual data points beyond the whiskers are plotted as individual points.

## `Synthetic Minority Over-sampling Technique ( SMOTE )`
The Synthetic Minority Over-sampling Technique is a sampling technique designed to address the issue of `Class Imbalance` in machine learning datasets.
Objective: The main objective of this sampling technique is to balance the class distribution by generating synthetic samples for the minority class ( The class having less number of samples )
Now the question is 
#### What is the Class imbalance and Why do we need to balance classes in the target variable?
Class Imbalance is a situation where the distribution of class in the target variable is NOT equal. In simple words, The number of the data points of one class in the target variable is less or more than the number of data points of another class. 

For Instance, let us consider a Binary classification problem where we need to predict whether an email is SPAM or NOT SPAM. if we have 90% NOT SPAM emails and 10% SPAM emails in the target variable of the dataset then we can say that it is a problem of Class Imbalance. NOTE: Here, SPAM is one class and NOT SPAM is another class.

When we train the machine learning model with this Class Imbalance dataset, the `Model tends to be more biased towards the majority class`. `The model may NOT perform on minority class` which leads to `Overfitting` with `High Accuracy`. 

So, that is the reason for Why do we need to address the class imbalance in datasets.

## `Confusion Matrix`
It is an important tool used in machine learning to evaluate the performance of the classification model. Even though we have `accuracy_score` to evaluate the classification model, All the ML practitioners use this confusion matrix because it will give us a comprehensive and detailed summary of how well the classification model is performing. this will give us a comprehensive and detailed summary of the different types of errors that the model is making. Here the types of errors could be `False Positive ` and `False Negative`
This confusion matrix helps in Decision Making as well. Decision-making in the sense, of making informed decisions about which model is the best suited for the particular problem if we use different models. Accuracy score alone is not a good practice used for classification model evaluation. The confusion matrix helps us better understand how well the model is performing in each class.


              Predicted values
                1       0                      

        1       100     50 

        0       20      89        

`True Positive`: Number of instances where the Predicted class is Positive (class 1) as well as the Actual class is Positive ( class 1). ie. The model Correctly (True) Predicted the Positive class. `NO ERROR`


`False Positive`: Number of instances where the Predicted class is Positive (class 1) but the Actual class is Negative (class 0). ie. The model is Wrongly(False) Predicted the positive class. `YES ERROR`

`True Negative`: Number of instances where the Predicted class is Negative (class 0) as well as the Actual class is Negative (class 0). ie. The model is Correctly (True) Predicted the Negative class. `NO ERROR`

`False Negative`: Number of instances where the Predicted class is Negative (class 0) but the Actual class is Positive (class 1). ie. The model is Wrongly (False) Predicted the Negative class. `YES ERROR`

      True Positive = 100
      False Positive = 20
      True Negative = 89
      False Negative = 50

#### `Precision`
Precision for class 1 = Out of All predicted class 1 instances, How many of them are actually class 1 

Precision for class 0 = Out of All predicted class 0 instances, How many of them are actually class 0

#### `Recall or Sensitivity or True Positive Rate`

I have a curiosity to "Measure the ability of the modal to predict the correct positive class"


            predicted class           Actual positive class
                  1                             1
                  1                             1
                  1                             1
                  1                             1
                  1                             1
                  1                             1
                  0                             1
                  0                             1
                  0                             1
                  0                             1

      True Positives (TP) = 6
      False Negatives (FN) = 4
Here, Total number of Actual Positive classes  = True Positives (TP) + False Negatives (FN)

Now, `What is the probability that the  model predicts correct positive class ?`

            A = Total number of Instances where the model correctly predicts positive class = True Positives (TP) = 6
            B = Total number of Actual Positive classes = True Positives (TP) + False Negatives (FN) = 10

            
$$ probability = \frac{A}{B} = \frac{6}{10} $$

$$ or $$

$$ Recall = Sensitivity = True Positive Rate = \frac{A}{B} = \frac{6}{10} $$

That means, The model that we have trained for the Binary classification problem, is working 60% correctly for predicting the positive class.

Key point: `Hight recall or sensitivity or True Positive Rate indicates that the model is Good at predicting positive classes`


#### F1 score

F1 score is the harmonic mean of precision and Recall 

     let ,      A = Precision 
                B = Recall

$$ \frac{1}{F1} = \frac{1}{2}[\frac{1}{A} + \frac{1}{B}] $$

$$ \frac{1}{F1} = \frac{2(A + B)}{AB} $$

$$ F1 = \frac{2AB}{A + B} $$


 
## `ROC curve`
The ROC curve is a Graphical representation of the performance of the Binary classification problem at various threshold settings.it is plots `True Positive Rate`( Sensitivity ) against `False Positive Rate`( Specificity ) for different threshold values  


# Maximum Likelihood Estimation
Maximum Likelihood Estimation is a statistical method used in Machine learning to find the values of the parameters that maximize the likelihood function


## GridSearchCV

`GridSearchCV` is a class provided by sklearn's `model_selction` module in Python. GridSeachCV helps us to find the best hyperparameters that allow the machine learning models to improve the performance  

## `Decision Tree Algorithm`

The `Linear Regression` and `Logistic Regression` are fundamental supervised machine learning algorithms used for `regression` and `classification` respectively. Likewise, a Decision tree is one of the most popular and widly used Non-parametric supervised learning algorithms `used for both regression and classification`. The name itself suggests that it is a tree-like structure with `Root Node`, `Branches`, `Decision  Nodes / Internal Nodes`, and `Leaf Nodes`.

* Nodes represent Features
* Branches represent Decision rule
* Leaf Nodes represent the output class / the class label

#### The decision tree algorithm learns from data and makes decisions(predictions) by `splitting the data based on the values of the features`



![image](https://github.com/user-attachments/assets/8d396ea4-7dc7-4199-803c-8a575ede1117)


Now, The very first question that comes to our mind after defining the decision tree is this:

#### `How do we build a decision tree algorithm such that the machine can learn the patterns from the training data and then make a prediction on test data? `

