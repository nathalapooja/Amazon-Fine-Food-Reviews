#################################################
Dataset-Amazon Fine Food Reviews
#################################################

On PC:
Required libraries and packages:
1)pip
step to install: sudo apt-get install python-pip
2) NLTK library - NLTK 3.2.1
steps to install: a)$python
                  b)import nltk
                  c)nltk.download('stopwords')
                  d)nltk.download('wordnet')
3) scikit - Latest
steps to install: a)sudo pip install scikit
3) Numpy - Latest 
steps to install: a)sudo pip install numpy
4) Pandas - Latest
steps to install: a)sudo pip install pandas

System configrationns to run:
a) download spark-2.2.0-bin-hadoop2.7.tgz zip file from google
b)tar -xvzf spark-2.2.0-bin-hadoop2.7.tgz
c)sudo mv spark-2.2.0-bin-hadoop2.7 ~/
d)go to spark-2.2.0-bin-hadoop2.7 folder from home
e)go to conf folder inside it and change the spark.driver.memory in spark-defaults.conf to 8g if your system RAM is more than 8G. Press Cntrl+S and save it



Steps to run:

1) Download the zip folfer and unzip it.The folder has all code files, input user reviews file
2) Following are the names of the files and their actions-
	NaiveBAccuracy.py - Naive Bayes implimentation from scratch, prints the accuracy for alpha value 1 to 5
	NaiveBSklearn-lib.py - Naive Bayes algorithm implemented using scikit learn library
	NaiveBInput-user.py - Naive Bayes algorithm to fit the user input reviews from input file to the model and predict the rating of the reviews with alpha value 2 and writing these user reviews along with outputs to output file

	KNNAccuracy.py - K Nearest Neighbor implimentation from scratch for K value between 10 to 50 with stepsize of 10
	KNNSklearn-lib.py - K Nearest Neighbor algorithm implemented using scikit learn library
	KNNInput-user.py - K Nearest Neighbor algorithm to fit the user input reviews from input file to the model and predict the rating of given input reviews with K  value 25 and writing these user reviews along with outputs to output file
        LogisticAccuracy.py- Logistic regression implementation from scratch, prints the accuracy of the predictions made by the algorithm
        LogisticSklearn.py-Logistic Regression implemented using the scikit learn library

3) To run the file, follow the below command
	
	spark-submit <filename>

	Example: spark-submit NaiveBAccuracy.py

Important Note:
1) Make sure that the dataset file is named as Reviews.csv
2) Reviews.csv file should be present in the same directory as the pyspark file
3) Input.txt has the user input reviews which should be in same directory for which ratings are predicted.
4) Output file will be stored in the same directory of code files and inputs

As the dataset taken have size above 300 MB we used an amazon EC2 instance for running this huge dataset.
We have create Amazon Ec2 Instance:
Steps to create and launch Amazon EC2 instance.
1.) go to AWS console choose launch instance and select ubuntu 64 bit operating system
2.) Choose instance Type t2.2xlarge general purpose with 32GB memory and click on create
3.) Make sure the instance is running and download the key file to desktop

To connect to the instance type:
4.)cd Desktop
5.)chmod 400 <keyfile>
6.)ssh -i “<keyfile>.pem" ubuntu@ec2-52-14-246-18.us-east-2.compute.amazonaws.com
7.)Follow above steps to install required packages to run project.
8.) Finally, use FileZilla to connect to EC2 instance for file transfer between your server and desktop.




