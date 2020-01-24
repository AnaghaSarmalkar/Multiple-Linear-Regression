Code for calculating linear regression coefficients with and without gradient descent has been implemented.
Implementation of Gradient Descent has been explained in the comments of LinearRegression.py

Please follow the following steps to implement.

- Load the input files and the LinearRegression.py the on aws s3 bucket.

- Load these files on the local system of hadoop
	aws s3 cp s3://anaghacloudassign3/LinearRegression.py .
	aws s3 cp s3://anaghacloudassign3/yxlin.csv .
	aws s3 cp s3://anaghacloudassign3/yxlin2.csv .

- Make a folder on HDFS to store the input files
	hadoop fs -mkdir /assignment4
	hadoop fs -mkdir /assignment4/input

- Put the data files on HDFS
	hadoop fs -put yxlin.csv /assignment4/input
	hadoop fs -put yxlin2.csv /assignment4/input

- Run the LinearRegression on the input files. Provide 4 arguments where 
	arg[0]: LinearRegression.py
	arg[1]: Path of input .csv file placed on hadoop (/assignment4/input/yxlin.csv)
	arg[2]: Path of uncreated folder for output of coefficients.

	spark-submit LinearRegression.py /assignment4/input/yxlin.csv /assignment4/output1

- The output for both is also printed on the console. Console images have been attached.

- Collect the output for output of coefficients without gradient descent.
	hadoop fs -cat /assignment4/output1/* > yxlin.out.txt

- Download the output to the aws s3 bucket.
	aws s3 cp ./yxlin.out.txt s3://anaghacloudassign3/yxlin.out.txt

- Repeat the same for yxlin2.csv


