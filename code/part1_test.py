import string
import unittest
import a1_preproc
from a1_preproc import *
import html
import re

class RegExAbbrPunctTests(unittest.TestCase):
    def test_basicReg(self):
        test = "Hey..."
        self.assertEqual(preproc1(test, [4]), "Hey ...")
    def test_abbr(self):
        test = "e.g."
        self.assertEqual(preproc1(test, [4]), "e.g.")

    def test_abbr_punct(self):
        test = "e.g..."
        self.assertEqual(preproc1(test, [4]), "e.g. ..")

    def test_abbr_punct_multi(self):
        test = "e.g.;etc.!!Mr.TheDog's"
        self.assertEqual(preproc1(test, [4]), "e.g. ; etc. !! Mr. TheDog's")

class Preproc5Tests(unittest.TestCase):
    def test_split_clitic(self):
        test = "She's the man"
        self.assertEqual(preproc1(test, [4,5]), "She 's the man")


class Preproc5Tests(unittest.TestCase):
    def test_basic(self):
        test = "firing/NN"
        self.assertEqual(preproc1(test, [5]), "fire/NN")


class RegExCliticTests(unittest.TestCase):
    def test_clitic(self):
        test = "Dog's Dogs'"
        self.assertEqual(preproc1(test, [5]), "Dog 's Dogs '")
    # note theres a problem with the clitics


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
    def test_wordSpacerAlreadySpaced(self):
        test = "Hello World Hi"
        self.assertEqual(wordSpacer(6, 10, test), "Hello World Hi")

class RegexPunctuationTests(unittest.TestCase):
    def test_regexabbr(self):
        text = "e.g.|etc.|[.,']*"
        test = "e.g."
        #re = re.compile(text)
        self.assertEquals(re.sub(text, "", test), "")

    def test_regexabbrwithmultipunct(self):
        text = "e.g.|etc.|([.,'])*"
        test = "e.g. ... etc.."
        #re = re.compile(text)
        self.assertEquals(re.sub(text, "fire", test), "fire fire firefire")

    def test_regexabbrwithmultipunctcombined(self):
        text = "e.g.|etc.|([.,'])+"
        test = ",,e.g.... etc.."
        # re.compile(text)
        self.assertEquals(re.sub(text, "fire", test), "firefirefire firefire")

    #def test_regexmatch(self):



class RegexHtmlTests(unittest.TestCase):
    def test_basicURL(self):
        url = "www.hello.com"
        self.assertEqual(removeUrlFunction(url), "")

    def test_HTTPSUrl(self):
        url = "https://google.com"
        self.assertEqual(removeUrlFunction(url), "")

    def test_sentenceURLEmbedded(self):
        url = "hellowww.hello.comWorld"
        self.assertEqual(removeUrlFunction(url), "hello")
 
    def test_sentenceURLSpaced(self):
        url = "hello www.hello.com World"
        self.assertEqual(removeUrlFunction(url), "hello World")

    def test_sentenceURLwithNon_urlPunctuation(self):
        url = "hello www.hello.com,World"
        self.assertEqual(removeUrlFunction(url), "hello ,World")

    def test_senetenceURLwith_punctuation(self):
        url = "this is an url https://www.google.com/?search=comma,works end"
        self.assertEqual(removeUrlFunction(url), "this is an url end")


    def test_sentenceURLwith_slashandletterafter(self):
        url = "this is an url https://www.google.com/search=comma,works end"
        self.assertEqual(removeUrlFunction(url), "this is an url end")

    def test_sentence_multiple_TLD(self):
        url = "www.google.com.gz.biz/search=comma,works end"
        self.assertEqual(removeUrlFunction(url), "end")

    def test_sentence_multiple_slashes(self):
        url = "this is https://www.google.com.biz.gz/search=comma/?=fire,workd end"
        self.assertEqual(removeUrlFunction(url), "this is end")


class HTMLRemovalTests(unittest.TestCase):
    def test_htmlTagremoval(self):
        text = "&gt;New Page &#62;"
        self.assertEqual(preproc1(text, [2]), ">New Page >")


    """def testRemoveSomePunct(self):

        test = "}He@ll.oW/or]ld^"
        print(preproc1(test, [4]));
        self.assertEqual(preproc1(test, [4]), "HelloWorld") """

class PartOneStepTwoTestCases(unittest.TestCase):
    def testRemoveSomePunct(self):
        test = "}He@ll.oW/or]ld^"
        #print(preproc1(test, [4]));
        #self.assertEqual(preproc1(test, [4]), "HelloWorld")"""


if __name__ == "__main__":
    print(string.punctuation)
    unittest.main()
