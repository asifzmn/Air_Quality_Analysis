from itertools import groupby
import unittest


def make_word_to_count(words):
    """
    Return a dictionary that maps each word in words to the number of
    times it appears in words.

    This implementation uses itertools.groupby.

    # >>> res = make_word_to_count(['csc', 'a08', 'is', 'csc', 'great', 'is', 'csc'])
    # >>> res == {'csc': 3, 'a08': 1, 'is': 2, 'great': 1}
    True
    """
    # Sort the words to group identical items together
    sorted_words = sorted(words)
    # Group by word and count occurrences
    word_to_count = {key: len(list(group)) for key, group in groupby(sorted_words)}
    return word_to_count


class TestMakeWordToCount(unittest.TestCase):
    """Test cases for function make_word_to_count."""

    def test_provided_example(self):
        actual = make_word_to_count(['csc', 'a08', 'is', 'csc', 'great', 'is', 'csc'])
        expected = {'csc': 3, 'a08': 1, 'is': 2, 'great': 1}
        msg = ("When we called make_word_to_count with argument:\n"
               + "['csc', 'a08', 'is', 'csc', 'great', 'is', 'csc']\n"
               + "we expected to see:\n" + str(expected)
               + "\nbut saw this instead:\n" + str(actual))
        print("test_provided_example")
        print(expected, actual, msg)
        self.assertEqual(expected, actual, msg)

    def test_does_not_modify_input(self):
        arg = ['csc', 'a08', 'is', 'csc', 'great', 'is', 'csc']
        original = ['csc', 'a08', 'is', 'csc', 'great', 'is', 'csc']
        make_word_to_count(arg)
        msg = ("The function make_word_to_count should not modify its input.\n"
               + "When we called it with argument:\n"
               + str(original)
               + "the input changed to:\n"
               + str(arg))
        print("test_does_not_modify_input")
        print(original, arg, msg)
        self.assertEqual(original, arg, msg)



if __name__ == "__main__":
    # print( make_word_to_count(['csc', 'a08', 'is', 'csc', 'great', 'is', 'csc']))
    unittest.main()

