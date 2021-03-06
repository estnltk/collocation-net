import pytest
from collocation_net.base_collocation_net import BaseCollocationNet

cn = BaseCollocationNet()


def test_unknown_word():
    with pytest.raises(ValueError):
        index = cn.row_index('testsõna')


def test_predict():
    pred = cn.predict_column_probabilities("kohv", ["tugev", "kange"])
    assert isinstance(pred[0][0], str)
    assert isinstance(pred[0][1], float)
    assert pred[0][0] == "kange"


def test_possible_collocates_output():
    nouns = cn.rows_used_with("kange", 5)
    adjectives = cn.columns_used_with("kohv", 5)
    assert len(nouns) == 5
    assert len(adjectives) == 5


def test_similar_nouns():
    similar_nouns = cn.similar_rows("kohv", 1)
    assert similar_nouns[0] in cn.rows


def test_similar_adjectives():
    similar_adjectives = cn.similar_columns("kange", 1)
    assert similar_adjectives[0] in cn.columns


def test_topic():
    topic = cn.topic("kohv")
    assert isinstance(topic, list)
    assert "kohv" in topic


def test_characterisation():
    char = cn.characterisation("kohv", 3, 5)
    assert isinstance(char, list)
    assert len(char) == 3

    first_topic = char[0]
    assert isinstance(first_topic, tuple)

    assert isinstance(first_topic[0], float)
    assert isinstance(first_topic[1], list)
    assert len(first_topic[1]) == 5


def test_predict_for_several():
    pred = cn.predict_for_several_rows(["kohv", "tee"], 1)
    assert len(pred) == 1
    pred = pred[0]
    assert isinstance(pred[0], str)
    assert isinstance(pred[1], float)


def test_predict_topic_for_several():
    pred = cn.predict_topic_for_several_rows(["kohv", "tee"], 1)
    assert len(pred) == 1
    pred = pred[0]
    assert isinstance(pred[0], list)
    assert isinstance(pred[1], float)
