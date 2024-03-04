# N19DCCN019 - Phạm Dung Bắc
# N19DCCN114 - Hoàng Hoài Nam
# N19DCCN128 - Lê Bá Minh Nhật

import numpy as np
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




stop_words = set(stopwords.words('english'))


def load_data(path):
    # Đọc dữ liệu và loại bỏ phần tử cuối
    # [' 11429 pattern detection and recognition  both processes have been carried out on an ibm computer which was programmed to simulate a spatial computer  the programs tested included the recognition process for reading handlettered sansserif alphanumeric characters    ']
    with open(path, 'r') as f:
        data = f.read().lower().replace("\n", " ").split("/")[:-1]

    result = []
    for doc in data:
        # Loại bỏ khoảng trắng đầu và cuối câu
        doc = doc.strip()

        # Index của khoảng trắng đầu tiên
        index_id = doc.find(" ")

        # Lấy từ khoảng trắng đầu câu đến hết, nhằm loại bỏ số đầu câu
        # ['pattern detection and recognition  both processes have been carried out on an ibm computer which was programmed to simulate a spatial computer  the programs tested included the recognition process for reading handlettered sansserif alphanumeric characters']
        doc = doc[index_id + 1:]
        result.append(doc)

    return result


def preprocess_doc(doc):
    # Loại bỏ ký tự đặc bi và chuyển thành chữ hoa thành chữ thường, word_tokenize để tách câu thành từ riêng lẻ
    tokens = word_tokenize(re.sub(r'[^\w\s]', '', doc).lower())

    # Loại bỏ stopword
    result = filter(lambda token: token not in stop_words, tokens)

    return list(result)


def generate_tfidf_matrix_documents(docs, all_counters, unique_words):
    # Đếm tần số xuất hiện của các từ trong mỗi câu
    #  Counter({'recognition': 2, 'computer': 2, 'pattern': 1,
    # 'detection': 1, 'processes': 1, 'carried': 1, 'ibm': 1, 'programmed': 1, 'simulate': 1, 'spatial': 1,
    # 'programs': 1, 'tested': 1, 'included': 1, 'process': 1, 'reading': 1, 'handlettered': 1, 'sansserif': 1,
    # 'alphanumeric': 1, 'characters': 1})
    counters_doc = list(map(Counter, docs))

    # Khởi tạo ma trận shape (11429, 12072), dòng là từng câu trong documents, cột là từ trong từ điển
    matrix = np.zeros((len(docs), len(unique_words)))

    # 'hibbard': 12067 key là từ, value là id
    word_dict = {word: i for i, word in enumerate(unique_words)}

    # i là id của từng câu 0 - 11428
    for i, counter in enumerate(counters_doc):
        # [7006, 4366, 2381, 10176, 7233, 2530, 8898, 982]
        # trả về array id của các từ trong từng doc
        word_indices = [word_dict[word] for word in counter.keys()]

        # {"computer": 2, "science": 3, "data": 1}, thì mảng numpy tf sẽ có giá trị [2, 3, 1].
        tf = np.fromiter(counter.values(), dtype=int)

        # Tính trọng số w_tf = 1 + log(tf)
        w_tf = 1 + np.log10(tf)

        matrix[i][word_indices] = w_tf

    N = len(documents)

    # Tạo vector biểu diễn idf
    # DF là số câu chứa từ đó
    # {"computer": 2, "science": 3, "data": 1}, thì mảng numpy df sẽ có giá trị [2, 3, 1].
    df = np.fromiter((all_counters[word] for word in unique_words), dtype=int)

    # Tính idf bằng log(N/df)
    idf = np.log10(N / df)

    # Tạo ma trận tf-idf = w_tf * idf
    for i in range(matrix.shape[0]):
        matrix[i] *= idf

    return idf, matrix


def cosine_similarity(queries_tfidf, matrix):
    # Chuyển queries_tfidf thành vector 1 chiều
    queries_tfidf = queries_tfidf.flatten()

    # Tính độ dài của queries_tfidf và ma trận tf-idf
    norm_query = np.linalg.norm(queries_tfidf)
    norm_matrix = np.linalg.norm(matrix, axis=1)

    # Tính tích vô hướng giữa queries_tfidf và mỗi vector document trong ma trận tf-idf
    dot_products = np.dot(matrix, queries_tfidf)

    # Chuẩn hóa giá trị tích vô hướng bằng cách chia cho tích độ dài vector query và vector document
    cosine_similarities = dot_products / (norm_query * norm_matrix)

    # Sắp xếp các giá trị cosine similarity từ cao đến thấp và trả về chỉ số của các document tương ứng
    sorted_indices = np.argsort(cosine_similarities)[::-1]

    return cosine_similarities, sorted_indices


# Tính TF_IDF cho từng câu query dựa vào bộ từ vựng và idf của văn bản
def genarate_tfidf_matrix_query(query, unique_words, idf):
    # Đếm số lần xuất hiện của từ trong câu query
    counter = Counter(query)

    queries_tfidf = np.zeros(len(unique_words))

    for word in counter.keys():
        # kiểm tra từ đó có trong unique_words hay không
        if word in unique_words:
            # Tính w_tf của query = 1 + log(tf)
            queries_tfidf[unique_words.index(word)] = 1 + np.log10(counter[word])

    # Trả về numpy tf_idf = w_tf * idf
    return queries_tfidf * idf


def process_data(data):
    return list(map(lambda doc: preprocess_doc(doc), data))


def search(queries, unique_words, idf):
    print("Tạo ma trận TF_IDF từ queries...")

    for i in range(len(queries)):
        queries_tfidf = genarate_tfidf_matrix_query(queries[i], unique_words, idf)

        scores, indicies = cosine_similarity(np.array(queries_tfidf), np.array(matrix_tfidf))
        print("Query là", i + 1, ":", queries[i])
        print("Kết quả tốt nhất (doc", indicies[0] + 1, ",", "score", scores[indicies][0], "):", documents[indicies[0]])
        print("Top 5:", indicies[:5] + 1)
        print("\n/\n")


if __name__ == "__main__":
    # Load dữ liệu và xử lý dữ liệu documents
    documents_data = load_data("./doc-text")
    documents = process_data(documents_data)

    # Load dữ liệu và xử lý dữ liệu query
    queries_data = load_data("./query-text")
    queries = process_data(queries_data)

    # Đếm tần suất xuất hiện của các từ trong toàn bộ tập văn bản
    all_counters = Counter(word for doc in documents for word in doc)

    # Lấy list các từ riêng biệt, gồm 12072 từ riêng biệt
    unique_words = list(all_counters.keys())

    print("Tạo ma trận TF_IDF từ documents...")

    idf, matrix_tfidf = generate_tfidf_matrix_documents(documents, all_counters, unique_words)

    print("Bắt đầu search...")
    search(queries, unique_words, idf)

