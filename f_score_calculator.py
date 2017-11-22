from sklearn.metrics import precision_recall_fscore_support
import xml_reader

# Global Variable
AVERAGE_TYPE = "micro"

input_token_data, output_label = xml_reader.load_data('data/ABSA16_Restaurants_Train_SB1_v2.xml')
pos_tag = {"O": 1, "TARGET-B": 2, "TARGET-I": 3}
encoded_output_data = []
file_name = "result_iter_10.txt"

for tag_data in output_label:
    output_data = [pos_tag[tag] for tag in tag_data]
    encoded_output_data.append(output_data)

test_data_set = encoded_output_data[1315:]

y_true = []
y_pred = []

with open(file_name) as f:
    pred_data_set = f.readlines()[::3]

    for test_data, pred_data in zip(test_data_set, pred_data_set):
        for test_ele, pred_ele in zip(test_data, pred_data):
            if test_ele != 1:
                y_true.append(test_ele)
                y_pred.append(int(pred_ele))

match_cnt = 0
for true, pred in zip(y_true, y_pred):
    if true == pred:
        match_cnt += 1

print(precision_recall_fscore_support(y_true, y_pred, average=AVERAGE_TYPE))

