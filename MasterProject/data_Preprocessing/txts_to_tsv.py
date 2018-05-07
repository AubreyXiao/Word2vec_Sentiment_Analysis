from os import listdir
import pandas as pd

negative ='/Users/xiaoyiwen/Desktop/datasets/train/neg'
positive ='/Users/xiaoyiwen/Desktop/datasets/train/pos'
unlabeled = '/Users/xiaoyiwen/Desktop/datasets/train/unsup'

#convert_files_to_tsv
def convert_files_to_tsv(tsv_filename,sentiment,directory):
    tsv_file = open(tsv_filename, 'w')
    tsv_file.write("id\tsentiment\treview\n")

    count = 0

    for filename1 in listdir(directory):

      if(filename1.endswith('.txt')==False):
        continue

      id = filename1.strip(".txt")
      path = directory +'/'+filename1
      file1 = open(path)
      review = file1.read()
      tsv_file.write(str(id)+"\t"+str(sentiment)+"\t"+str(review)+"\n")
      count = count+1

    return count

#convert_files_to_tsv
def convert_unsup_to_tsv(tsv_filename,directory):
    tsv_file = open(tsv_filename, 'w')
    tsv_file.write("id\treview\n")

    count = 0

    for filename1 in listdir(directory):

      if(filename1.endswith('.txt')==False):
        continue

      id = filename1.strip(".txt")
      path = directory +'/'+filename1
      file1 = open(path)
      review = file1.read()
      tsv_file.write(str(id)+"\t"+str(review)+"\n")
      count = count+1
      file1.close()
    return count

#show
def read_tsv(tsv_filename):
    tsv = pd.read_csv(tsv_filename,header=0,delimiter='\t',quoting=3)

    return tsv


