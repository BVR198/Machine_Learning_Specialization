## Machine-Learning-Concepts

## `Gradient Discent Algorithm`
- It is an optimization algorithm used in not only Machine learning but also in the Deep learning to `minimize the error ( cost function or loss function )` associated with the  model tha we are trained. it is a technique used for finding the best model's parameters Wieghts and Biases in order to reduce ( minimize ) the cost function or loss function.
-  ### How does this Work, I mean What is the actual work behind the Gradient Discent Algorithm ?
   Let us train the model with the Linear function f(x[i]) for Linear Regression problem and we have the cost function $J(w,b)$ which need to minimize by adjusting (w, b)

 $$f(x^{(i)}) = wx^{(i)} + b$$
 

 As we know that cost function $J(w,b)$ is a measure of the error between the predicted value $f(x^{i)})$ and actual value $y^{(i}) over all the training values $x^{(i)}, y^{(i)}$

 $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f(x^{(i)}) - y^{(i)})^2$$
 
Now, Our main objective is to find the best values of (w,b) such that the value of the cost function $J(w,b)$ will be minimum. In this Algorithm what we are doing is , we are changing the values of w, b untill we get the minimum value of the $J(w,b)$

updating the vales of the parameter w, b for each ieration using the mathematical equation :

$$ w = w - \alpha\frac{\partial J(w,b)}{\partial w}$$

$$ b = b - \alpha\frac{\partial J(w,b)}{\partial b}$$



Here,Gradient term represents the `Magnitude` and `Direction`. Basically, `it tells us How to chnge the parameters w, b to reduce the cost` .Where as $\alpha$   is `Learning Rate` Which is Hyperparameter determins the size of the step that we need to take during each iteration in order to get minima of the cost function
- Laregr leaning rate : A larger learning rate can cause the algorithm to `Converge more quickly`, but it `Oversoot` the minimum value of the cost function
- Smallar learning rate : on other hand , a smaller learning rate may `Converge more slowly` but `potentially more precision` 


## `One-Hot Encoding`

The Main purpose is to make easy for Machine Learning Algorithms beacause machine can understand only numerical values
One-hot Encoding is a technique used in Machine learning and data preprocessing to convert the `categorical` variables into `Numerical` variables (Binary numbers) 0s and 1s, where each category becomes a seperate binary feature. Each seperated features represents the presence(1) or obsence(0) of the particular category

## `Feature Engineering`
Feature Engineering is the process of `Creating new features` using the existing one or `Modifying the existing features` of the dataset to `improve the performance of a machine learning model` 


## `Outliers`
Before looking for `Outliers`, Let us first address the quetions like:
#### What are Outliers abd Why do we need to remove them ? 

`Outliers` are Nothing, But the points that are significatlty different from the majority of the data in a dataset. They can be usually high or low values that deviates from majority of the data distributionn. Now the second question is Why do we need to remove the outliers ? We need to remove the Outliers beacuase they can significantly influence the statistical measures like `Mean` and `Standard Deciation`. As we all know that the Mean is calculated by summingup all the values in the dataset and dividing by the total number of values. Outliers contribution is also there in calculation of the mean . `if there are high outliers, the mean tend to be higher than the  central tendency of the majority of the data`. Conversly, if there are lower outliers, the mean tends to be loewr than the central tendency of the majority of the data. `Standard Deviation` measures the amount of dispersion in the dataset . Outliers can increase the standard deviation because they contribute to greater variability.leading to an inflated standard deviation

#### How Can we Identify the Potenstial Outliers from the box plots ?

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
Synthetic Minority Over-sampling Technique is a sampling technique designed to address the issue of `Class Imbalance` in machine learning datasets.
Objective : The main ojectuve of this sampling technique is to balance the class distribution by generating the synthetic samples for the minority class ( The class having less number of samples )
Now the Question is 
#### What is the Class imbalance and Why do we need to balance classes in the target variable ?
Class Imbalance is a situation where the distribution of class in target variable is NOT equal. In simple words, The number of the data points of one class in target variable less or more than the number of data points of other class. 

For Instance, let us consider a Binary classification problem where we need to predict an email is SPAM or NOT SPAM. if we have 90% NOT SPAM emails and 10% SPAM emails in the target variable of the dataset then we can say that it is a problems of Class Imbalance. NOTE : Here, SPAM is one class and NOT SPAM is another class.

When we train the machine learning model with this Class Imbalance dataset, `Model tends to be more biased towards the majority class`. `The model may NOT perform on minority class` which leads to `Overfitting` with `High Accuracy`. 

So, that is the reason for Why do we need to address the class imbalance in datasets.

## `Confusion Matrix`
It is a important tool used in the machine learning to evaluate the performance of the classification model. Even though we have `accuracy_score` to evaluate the classification model, All the ML practitioners uses this confusion matrix because it will gives us a comprehensive and detailed summary of how well the classification model is performing.this will gives us comprehensive and detailed summary of different types of the errors that the model is making . Here the types of errors could be `False Positive ` and `False Negative`
This confusion matrix helps in Decision Making as well. Decision making in the sence, making an informed decisions about which model is best suite for particlar problem if we use different models. Accuracy score alone is not a good practice use for classification model evaluation. Cofusion matrix helps us in better understanding how well model is performing on each class.


              Predicted values
                1       0                      

        1       100     50 

        0       20      89        

`True Positive` : Number of instances where the Predicted class is Positive (class 1) as well as Actual class is Positive ( class 1). ie. The model Corectly (True) Predicted the Positive class. `NO ERROR`


`False Positive`: Number of instances where the Predicted class is Positive (class 1) but the Actual class is Negative (class 0). ie. The model is Wrongly(False) Predicted the positive class. `YES ERROR`

`True Negative`: Number of instances where Predicted class is Negative (class 0) as well as the Actual class is Negative (class 0). ie. The model is Correctly (True) Predicted the Negative class. `NO ERROR`

`False Negative`: Number of instances where Predicted class is Negative (class 0) but the Actual class is Positive (class 1). ie. The model is Wrongly (False) Predicted the Negative class. `YES ERROR`

      True Positive = 100
      False Positive = 20
      True Negative = 89
      False Negative = 50

## Recall or Sensitivity or True Positive Rate

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

That means, The model that we have trained for the Binary classification problem, is working $60%$ correct for predicting the positive class

      

 
## `ROC curve`
The ROC curve is a Graphical representation of the performance of the Binary classification problem at various threshold settings.it is plots `True Positive Rate`( Sensitivity ) against `False Positive Rate`( Specificity ) for different threshold values  










