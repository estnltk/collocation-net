import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class CollocationNetException(Exception):
    pass


class CollocationNet:
    """
    CollocationNet class
    """

    def __init__(self, model: str = 'LDA', data: str = "data/df.csv"):
        self.data = pd.read_csv(data, index_col=0)
        self.noun_dist = np.load(f"data/{model.lower()}_noun_distribution.npy")
        self.adj_dist = np.load(f"data/{model.lower()}_adj_distribution.npy")

        # with open(f"{model.lower()}_model.pickle", "rb") as f:
        #     self.model = pickle.load(f)

        with open(f"data/{model.lower()}_topics.pickle", "rb") as f:
            self.topics = pickle.load(f)

    def get_noun_index(self, word: str) -> int:
        for i, noun in enumerate(self.data.index):
            if noun == word:
                return i
        return -1

    def get_nouns_used_with(self, word: str, number_of_words: int = 10) -> list:
        # if word not in self.data.columns:
        #     raise CollocationNetException(f"Word '{word}' not in dataset")
        # return self.data[word].sort_values(ascending=False).index[:number_of_words].values.tolist()

        word_index = None

        for i, adj in enumerate(self.data.columns):
            if adj == word:
                word_index = i

        if word_index is None:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        adj_topic = self.adj_dist[:, word_index].argmax()
        noun_topic = self.noun_dist[:, adj_topic].argsort()[::-1][:number_of_words]

        return self.data.index[noun_topic].tolist()

    def get_adjectives_used_with(self, word: str, number_of_words: int = 10) -> list:
        # threshold ka (+ eelmisele) - lisada ise threshold, mis väärtusega minimaalselt võivad olla
        # sarnased sõnad vms, mõtle kas on pointi
        # if word not in self.data.index:
        #     raise CollocationNetException(f"Word '{word}' not in dataset")
        # return self.data.loc[word].sort_values(ascending=False).index[:number_of_words].values.tolist()

        word_index = self.get_noun_index(word)

        if word_index == -1:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        noun_topic = self.noun_dist[word_index].argmax()
        adj_topic = self.adj_dist[noun_topic].argsort()[::-1][:number_of_words]

        return self.data.columns[adj_topic].tolist()

    def get_similar_nouns(self, word: str, number_of_words: int = 10) -> list:
        knn = NearestNeighbors(n_neighbors=number_of_words + 1).fit(self.noun_dist)
        idx_of_word = None

        for i, noun in enumerate(self.data.index):
            if noun == word:
                idx_of_word = i

        if idx_of_word is None:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        neighbour_ids = knn.kneighbors(self.noun_dist[idx_of_word].reshape(1, -1), return_distance=False)
        neighbours = self.data.index.to_numpy()[neighbour_ids]

        return list(neighbours[0])[1:]

    def similar_noun_for_list(self, words: list) -> str:
        similar_for_each = {}

        for word in words:
            word_similar = self.get_similar_nouns(word, self.data.index.shape[0]-1)
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

        return sorted_common[0][0]

    def get_similar_adjectives(self, word: str, number_of_words: int = 10) -> list:
        adj_vals = self.adj_dist.T
        knn = NearestNeighbors(n_neighbors=number_of_words + 1).fit(adj_vals)
        idx_of_word = None

        for i, adj in enumerate(self.data.columns):
            if adj == word:
                idx_of_word = i

        if idx_of_word is None:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        neighbour_ids = knn.kneighbors(adj_vals[idx_of_word].reshape(1, -1), return_distance=False)
        neighbours = self.data.columns.to_numpy()[neighbour_ids]

        return list(neighbours[0])[1:]

    def similar_adjective_for_list(self, words: list) -> str:
        similar_for_each = {}

        for word in words:
            word_similar = self.get_similar_adjectives(word, self.data.columns.shape[0]-1)
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

        return sorted_common[0][0]

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
        TODO
        see tabel??
        :param word:
        :return:
        """
        word_index = self.get_noun_index(word)

        if word_index == -1:
            raise CollocationNetException(f"Word '{word}' not in dataset")

        topic_vector = self.noun_dist[word_index]
        sorted_index = topic_vector.argsort()[::-1][:number_of_topics]
        adjs_in_topics = pd.DataFrame(self.adj_dist, columns=self.data.columns)
        final_list = list()

        for i in sorted_index:
            final_list.append((round(topic_vector[i], 3), adjs_in_topics.loc[i].sort_values(ascending=False)[:number_of_adjs].index.tolist()))

        return final_list

    def predict(self, noun: str, adjectives: list) -> list:
        """
        TODO

        :param noun:
        :return:
        """
        characterisation = self.characterisation(noun, number_of_topics=self.noun_dist[0].shape[0])

        sorted_adjs = []

        for c in characterisation:
            for adj in c[1]:
                if adj in adjectives:
                    sorted_adjs.append(adj)
                    adjectives.remove(adj)

        return sorted_adjs

    def predict_for_several(self, words: str):
        char_for_each = {}

        for word in words:
            characterisation = self.characterisation(word, number_of_topics=self.noun_dist[0].shape[0])
            char_for_each[word] = characterisation

        sorted_for_each = {}

        for word, char in char_for_each.items():
            for c in char:
                for adj in c[1]:
                    if word not in sorted_for_each:
                        sorted_for_each[word] = [adj]
                    else:
                        sorted_for_each[word].append(adj)

        common = []

        for i, adjective in enumerate(sorted_for_each[words[0]]):
            is_in_all = sum([1 for l in words[1:] if adjective in sorted_for_each[l]])
            if is_in_all != len(words) - 1:
                continue

            c = [adjective, i]

            for l in words[1:]:
                c.append(sorted_for_each[l].index(adjective))

            common.append(c)

        sorted_common = sorted(common, key=lambda x: sum(x[1:]))

        if len(sorted_common) == 0:
            return None

        return sorted_common[0][0]

    def usable_phrase(self, noun: str, adj: str, number_of_topics: int = 10) -> str:
        """
        TODO
        NT KOLLANE ESMASPÄEV - ENNUSTA KAS ON OK VÕI EI
        vt seda järjest nt esimeses top klastris esmaspäev jaoks kui on

        võta näiteks top 10 klastrit ja siis iga klastri jaoks sellised omadussõnad, et
        väärtused oleks kõigil min 100x väiksemad suurimast vms, kui nende hulgas pole
        kuskil sõna, siis ei ole kasutatav, otherwise on

        :param word:
        :return:
        """
        word_index = self.get_noun_index(noun)

        if word_index == -1:
            raise CollocationNetException(f"Word '{noun}' not in dataset")

        adj_index = None

        for i, adj_in_df in enumerate(self.data.columns):
            if adj_in_df == adj:
                adj_index = i

        if adj_index is None:
            raise CollocationNetException(f"Word '{adj}' not in dataset")

        topic_vector = self.noun_dist[word_index]
        sorted_index = topic_vector.argsort()[::-1][:number_of_topics]
        adjs_in_topics = pd.DataFrame(self.adj_dist, columns=self.data.columns)

        for i in sorted_index:
            adjs = adjs_in_topics.loc[i][adjs_in_topics.loc[i] > 10].sort_values(ascending=False).index.tolist()
            for a in adjs:
                if a == adj:
                    return f"Fraas '{adj} {noun}' on koos kasutatav."

        return f"Fraas '{adj} {noun}' ei ole koos kasutatav."
