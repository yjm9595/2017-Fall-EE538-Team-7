from f1_measure import bio_classification_report


result_file_name = "cat_div_ver_add_data_epoch_30_hidden_128_filter_256_aaa.txt"

with open(result_file_name) as f:
    result_set = f.readlines()
    result_pred_set = result_set[::2]
    result_true_set = result_set[1::2]

    pred_set = []
    true_set = []

    for pred, true in zip(result_pred_set, result_true_set):
        for x in pred.strip():
            pred_set.append(x)

        for x in true.strip():
            true_set.append(x)

print(bio_classification_report(true_set, pred_set))
