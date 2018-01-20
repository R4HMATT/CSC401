import string
import unittest
import a1_preproc
from a1_preproc import *

class PartOneStepOneTestCase(unittest.TestCase):
    def testDelOneNewLines(self):
        test = "Hello\nWorld"
        self.assertEqual(preproc1(test, [1]), "HelloWorld")

    def testDelMultNewLines(self):
        test = "\nHello\nWor\nld"
        self.assertEqual(preproc1(test, [1]), "HelloWorld")

class WordSpacerTests(unittest.TestCase):

    def test_wordSpacerNormalCase(self):
        test = "HelloWorldHi"
        self.assertEqual(wordSpacer(5,9, test), "Hello World Hi") 
    def test_wordSpacerBeginningCase(self):
        test = "HelloWorldHi"
        self.assertEqual(wordSpacer(0, 4, test), "Hello WorldHi")
    def test_wordSpacerBeginningPlus1Case(self):
        test = "HelloWorldHi"
        self.assertEqual(wordSpacer(1, 4, test), "H ello WorldHi")
    def test_wordSpacerEndCase(self):
        test = "HelloWorldHi"
        self.assertEqual(wordSpacer(10, 11, test), "HelloWorld Hi")


class RegexHtmlTests(unittest.TestCase):
    def test_basicURL(self):
        url = "www.hello.com"
        self.assertEqual(removeUrlFunction(url), "")

    def test_sentenceURL(self):
        url = "hellowww.hello.comWorld"
        self.assertEqual(removeUrlFunction(url), "helloWorld")
 
""" def testRemoveSomePunct(self):
        test = "}He@ll.oW/or]ld^"
        print(preproc1(test, [4]));
        self.assertEqual(preproc1(test, [4]), "HelloWorld") """

"""class PartOneStepTwoTestCases(unittest.TestCase):
    def testRemoveSomePunct(self):
        test = "}He@ll.oW/or]ld^"
        #print(preproc1(test, [4]));
        #self.assertEqual(preproc1(test, [4]), "HelloWorld")"""


if __name__ == "__main__":
    print(string.punctuation)
    unittest.main()
