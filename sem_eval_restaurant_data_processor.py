from gensim.models import Word2Vec
import xml_reader
from nltk.corpus import stopwords


# Function
def create_word2vec_model(size=100, window=2, min_count=1, workers=4, iter=100, sg=1):
    input_token_data, output_label = xml_reader.load_data_for_task_3('data/ABSA16_Restaurants_Train_SB1_v2.xml')

    with open("data/ALL_TRAIN_DATA.data") as f:
        for line in f.readlines():
            input_token_data.append(line.split())

    review_model = Word2Vec(input_token_data, size=size, min_count=min_count, window=window, workers=workers, iter=iter, sg=sg)
    review_model.init_sims(replace=True)
    return review_model


def create_model_file(file_name="sem_eval_restaurant_model.vec"):
    (create_word2vec_model()).save(file_name)


if __name__ == "__main__":
    model = create_word2vec_model()
    print(model.most_similar("expensive"))
    print(model.most_similar("waiter"))
    print(model.most_similar("noon"))
    #
    create_model_file("sem_eval_restaurant_model_add_data.vec")
    print("FINISH!!!")


