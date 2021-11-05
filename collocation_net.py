import pickle
import scipy
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


class CollocationNetException(Exception):
    pass


class CollocationNet:
    """
    CollocationNet class
    """

    def __init__(self, collocation_type: str = 'noun_adjective', read_data: bool = False):
        path = f"{os.path.dirname(os.path.abspath(__file__))}/data/{collocation_type}"
        # path = f"data/{collocation_type}"
        if read_data:
            self.data = pd.read_csv(f"{path}/df.csv", index_col=0)
        self.noun_dist = np.load(f"{path}/lda_noun_distribution.npy")
        self.adj_dist = np.load(f"{path}/lda_adj_distribution.npy")
        self.nouns = np.load(f"{path}/nouns.npy", allow_pickle=True)
        self.adjectives = np.load(f"{path}/adjectives.npy", allow_pickle=True)

        self.adj_dist_probs = self.adj_dist / self.adj_dist.sum(axis=0)

        with open(f"{path}/lda_topics.pickle", "rb") as f:
            self.topics = pickle.load(f)

    def noun_index(self, word: str) -> int:
        for i, noun in enumerate(self.nouns):
            if noun == word:
                return i
        raise CollocationNetException(f"Word '{word}' not in the dataset.")

    def adjective_index(self, word: str) -> int:
        for i, adj in enumerate(self.adjectives):
            if adj == word:
                return i
        raise CollocationNetException(f"Word '{word}' not in the dataset.")

    def nouns_used_with(self, word: str, number_of_words: int = 10) -> list:
        word_index = self.adjective_index(word)

        adj_topic = self.adj_dist[:, word_index].argmax()
        noun_topic = self.noun_dist[:, adj_topic].argsort()[::-1][:number_of_words]

        return self.nouns[noun_topic].tolist()

    def adjectives_used_with(self, word: str, number_of_words: int = 10) -> list:
        word_index = self.noun_index(word)

        noun_topic = self.noun_dist[word_index].argmax()
        adj_topic = self.adj_dist[noun_topic].argsort()[::-1][:number_of_words]

        return self.adjectives[adj_topic].tolist()

    def similar_nouns(self, word: str, number_of_words: int = 10) -> list:
        knn = NearestNeighbors(n_neighbors=number_of_words + 1).fit(self.noun_dist)
        idx_of_word = None

        for i, noun in enumerate(self.nouns):
            if noun == word:
                idx_of_word = i

        if idx_of_word is None:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        neighbour_ids = knn.kneighbors(self.noun_dist[idx_of_word].reshape(1, -1), return_distance=False)
        neighbours = self.nouns[neighbour_ids]

        return list(neighbours[0])[1:]

    def similar_nouns_for_list(self, words: list, num_of_words: int = 10) -> list:
        similar_for_each = {}

        for word in words:
            word_similar = self.similar_nouns(word, self.nouns.shape[0] - 1)
            similar_for_each[word] = word_similar

        common = []

        for i, noun in enumerate(similar_for_each[words[0]]):
            if noun in words:
                continue

            is_in_all = sum([1 for l in words[1:] if noun in similar_for_each[l]])
            if is_in_all != len(words) - 1:
                continue

            c = [noun, i]

            for l in words[1:]:
                c.append(similar_for_each[l].index(noun))

            common.append(c)

        sorted_common = sorted(common, key= lambda x: sum(x[1:]))

        if len(sorted_common) == 0:
            return None

        similar_words = [sort[0] for sort in sorted_common]

        return similar_words[:num_of_words]

    def similar_adjectives(self, word: str, number_of_words: int = 10) -> list:
        adj_vals = self.adj_dist.T
        knn = NearestNeighbors(n_neighbors=number_of_words + 1).fit(adj_vals)
        idx_of_word = None

        for i, adj in enumerate(self.adjectives):
            if adj == word:
                idx_of_word = i

        if idx_of_word is None:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        neighbour_ids = knn.kneighbors(adj_vals[idx_of_word].reshape(1, -1), return_distance=False)
        neighbours = self.adjectives[neighbour_ids]

        return list(neighbours[0])[1:]

    def similar_adjectives_for_list(self, words: list, num_of_words: int = 10) -> list:
        similar_for_each = {}

        for word in words:
            word_similar = self.similar_adjectives(word, self.adjectives.shape[0] - 1)
            similar_for_each[word] = word_similar

        common = []

        for i, adjective in enumerate(similar_for_each[words[0]]):
            if adjective in words:
                continue

            is_in_all = sum([1 for l in words[1:] if adjective in similar_for_each[l]])
            if is_in_all != len(words) - 1:
                continue

            c = [adjective, i]

            for l in words[1:]:
                c.append(similar_for_each[l].index(adjective))

            common.append(c)

        sorted_common = sorted(common, key= lambda x: sum(x[1:]))

        if len(sorted_common) == 0:
            return None

        similar_words = [sort[0] for sort in sorted_common]

        return similar_words[:num_of_words]

    def topic(self, word: str) -> int:
        """
        TODO
        .....
        :param word:
        :param with_info:
        :return:
        """
        for topic, words in self.topics.items():
            if word in words:
                return words

        raise CollocationNetException(f"Word '{word}' not in dataset")

    def characterisation(self, word: str, number_of_topics: int = 10, number_of_adjs: int = 10):
        """

        :param word:
        :param number_of_topics:
        :param number_of_adjs:
        :return:
        """
        word_index = self.noun_index(word)

        if word_index == -1:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        topic_vector = self.noun_dist[word_index]
        sorted_index = topic_vector.argsort()[::-1][:number_of_topics]
        adjs_in_topics = pd.DataFrame(self.adj_dist, columns=self.adjectives)
        final_list = list()

        for i in sorted_index:
            final_list.append((round(topic_vector[i], 3), adjs_in_topics.loc[i].sort_values(ascending=False)[:number_of_adjs].index.tolist()))

        return final_list

    def predict(self, noun: str, adjectives: list = None, num_of_adjectives: int = 10) -> list:
        """
        TODO

        :param noun:
        :param adjectives:
        :param num_of_adjectives:
        :return:
        """
        noun_index = self.noun_index(noun)
        noun_topics = self.noun_dist[noun_index]

        avg_per_adj = np.matmul(noun_topics, self.adj_dist_probs)

        if adjectives is None:
            top_adj_ind = np.argpartition(avg_per_adj, -num_of_adjectives)[-num_of_adjectives:]
            top_adj_ind = top_adj_ind[np.argsort(avg_per_adj[top_adj_ind])][::-1]
            top_probs = avg_per_adj[top_adj_ind]
            top_adjs = self.adjectives[top_adj_ind]
            return list(zip(*(top_adjs, top_probs)))

        results = []

        for adj in adjectives:
            adj_idx = self.adjective_index(adj)

            if adj_idx == -1:
                continue

            results.append((adj, avg_per_adj[adj_idx]))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def predict_for_several(self, words: list, num_of_adjectives: int = 10):
        # avg_probs = defaultdict(lambda:1)
        avg_probs = defaultdict(int)

        for word in words:
            adj_probs = self.predict(word, num_of_adjectives=self.adjectives.shape[0])
            for adj, prob in adj_probs:
                avg_probs[adj] += prob

        avg_probs = [(k, v / len(words)) for k, v in avg_probs.items()]

        return sorted(avg_probs, key=lambda x: x[1], reverse=True)[:num_of_adjectives]

    def predict_topic_for_several(self, words: list, num_of_topics: int = 10, num_of_adjectives: int = 10):
        """
        When given a list of several nouns, it predicts the most likely topic for them and returns
        the average probability of that topic and the top adjectives describing that topic.
        :param words:
        :param num_of_topics:
        :param num_of_adjectives:
        :return:
        """
        avg_probs = defaultdict(int)

        for word in words:
            word_idx = self.noun_index(word)
            topic_probs = self.noun_dist[word_idx]
            for i, prob in enumerate(topic_probs):
                avg_probs[i] += prob

        avg_probs = [(k, v / len(words)) for k, v in avg_probs.items()]
        sorted_topics = sorted(avg_probs, key=lambda x: x[1], reverse=True)[:num_of_topics]
        predicted_topics = []

        for topic_id, topic_prob in sorted_topics:
            topic_adjs = self.adj_dist[topic_id]
            top_adj_ind = np.argpartition(topic_adjs, -num_of_adjectives)[-num_of_adjectives:]
            top_adj_ind = top_adj_ind[np.argsort(topic_adjs[top_adj_ind])][::-1]
            predicted_topics.append((list(self.adjectives[top_adj_ind]), topic_prob))

        return predicted_topics

    def usable_phrase(self, noun: str, adj: str, number_of_topics: int = 10) -> str:
        """

        :param noun:
        :param adj:
        :param number_of_topics:
        :return:
        """
        word_index = self.noun_index(noun)

        if word_index == -1:
            raise CollocationNetException(f"Word '{noun}' not in dataset")

        adj_index = None

        for i, adj_in_df in enumerate(self.adjectives):
            if adj_in_df == adj:
                adj_index = i

        if adj_index is None:
            raise CollocationNetException(f"Word '{adj}' not in dataset")

        topic_vector = self.noun_dist[word_index]
        sorted_index = topic_vector.argsort()[::-1][:number_of_topics]
        adjs_in_topics = pd.DataFrame(self.adj_dist, columns=self.adjectives)

        for i in sorted_index:
            adjs = adjs_in_topics.loc[i][adjs_in_topics.loc[i] > 10].sort_values(ascending=False).index.tolist()
            for a in adjs:
                if a == adj:
                    return f"Fraas '{adj} {noun}' on koos kasutatav."

        return f"Fraas '{adj} {noun}' ei ole koos kasutatav."
