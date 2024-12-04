import unittest

class TestInequality(unittest.TestCase):
    def test_not_equal_numbers(self):
        a = 5
        b = 10
        self.assertNotEqual(a, b)  # Passes because 5 != 10

    def test_not_equal_strings(self):
        str1 = "hello"
        str2 = "world"
        self.assertNotEqual(str1, str2)  # Passes because "hello" != "world"

    def test_not_equal_failure(self):
        x = 42
        y = 421
        self.assertNotEqual(x, y)  # Fails because 42 == 42

if __name__ == "__main__":
    unittest.main()