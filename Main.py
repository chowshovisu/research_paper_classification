import Read_Data
import Pre_Processing
import SVM_Classification
from sklearn.model_selection import train_test_split

def check_for_new_test(classifier):

    while 1:
        new_str = str(input('Give a sample input string to Test:'))

        if new_str == '\n':
            break

        tokenized_input = Pre_Processing.tokenize_preprocess_corpus([new_str])

        predicted_output = classifier.predict(tokenized_input)

        print('The Prediction Is: {}'.format(predicted_output))

if __name__ == '__main__':

    # reads and processes the csv file
    csv_dataframe = Read_Data.read_csv(file_name='rt-polaritydata/Plus_800.csv', separator=',')
    reports, labels = Read_Data.process_data(data=csv_dataframe)

    unique_list = sorted(list(set(labels)))

    for item in unique_list:
        print('{} = {}'.format(item, labels.count(item)))

    # tokenize and pre-process reports
    tokenized_reports = Pre_Processing.tokenize_preprocess_corpus(reports)

    print(tokenized_reports[0])

    # # Divide the reports and labels into Training  and Test Documents

    train_reports,test_reports,train_labels,test_labels  = train_test_split(tokenized_reports, labels, test_size = 0.33, random_state=42)

    # Do the classification!!
    classifier=SVM_Classification.SVM_Normal(trainDoc=train_reports, trainClass=train_labels,
                                  testDoc=test_reports, testClass=test_labels, tfIdf=True)



    check_for_new_test(classifier)

