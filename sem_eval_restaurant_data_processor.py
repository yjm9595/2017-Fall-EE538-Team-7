from gensim.models import Word2Vec
import xml_reader
from nltk.corpus import stopwords

# Global Variable
STOP_WORDS = (stopwords.words('english')) \
             + ['&apos;t', '&apos;s', "&quot;", '...', '&apos;re', '&apos;ve', '.....'] \
             + list("!@#$%^&*()[]{};:,./<>?\|`~-=_+") \
             + [stop_word.title() for stop_word in stopwords.words('english')]


# Function
def create_word2vec_model(size=100, window=2, workers=4, iter=100, sg=1):
    review_lst = []

    for line in xml_reader.load_data()[0]:
        filtered_words = [word for word in line if word not in STOP_WORDS]
        review_lst.append(filtered_words)

    review_model = Word2Vec(review_lst, size=size, window=window, workers=workers, iter=iter, sg=sg)
    review_model.init_sims(replace=True)
    return review_model


def create_model_file(file_name="sem_eval_restaurant_model.vec"):
    (create_word2vec_model()).save(file_name)


if __name__ == "__main__":
    model = create_word2vec_model()
    print(model.most_similar("expensive"))
    print(model.most_similar("waiter"))

    create_model_file()


