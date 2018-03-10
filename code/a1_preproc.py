import time
import spacy
import string
import sys
import argparse
import os
import re
import json
#from html.parser import HTMLParser
import html
#print(html.__file__)




indir = '/u/cs401/A1/data/';

#~~~~~~~~~~~~~~~bunch of regex set up~~~~~~~~~

# original reg used in removeurl
url_reg = re.compile("(https://|www\.|http://)([a-zA-Z0-9])+(\.[a-zA-Z0-9]+)*(/([a-zA-Z0-9]?[%s]?)*)* *" % re.escape(string.punctuation))


# global for step 9
with open('/u/cs401/Wordlists/abbrev.english') as f:
    abbrev_list = f.read().split()

# create a regex to replaxe stopwords
with open('/u/cs401/Wordlists/StopWords') as f:
        stopwords = f.read()
stop_regex = (r"(?<!\S)(%s)/[\w%s]+(\$*|\b)" % (stopwords.replace("\n", "|")[:-1], re.escape(string.punctuation)))
stop_reg = re.compile(stop_regex, re.IGNORECASE)


# load spacy once
nlp = spacy.load('en', disable=['parser', 'ner'])

#taken from clitic split step
clit_reg = re.compile("(?='s|s'|'ve|'m|n't)")


#taken from step 4, we do it outside for efficiency
f = open(r"/u/cs401/Wordlists/abbrev.english");
abbr = f.read()
f.close();

abbr += ('\ne.g.\ni.e.\netc.')
#abbr = abbr.replace('\n', '|')
#abbr.append('e.g')
#abbr.append('i.e.')
#abbr.append('etc.')
abbr = abbr.split()
set_abbr = set(abbr)
abbr_regex = ""
for item in abbr:
    item1 = item.replace('.', '\.')
    abbr_regex += item1 + "|"
    #modify_abbr += item + "|"

punct = string.punctuation.replace("'", "")
punct = punct.replace('-', "")
#~~~some failed attempts below
#pure_abbr_reg = re.compile(("(?=(%s){1})" % abbr.replace("\n","|")), re.IGNORECASE)
#abbr_regex = abbr_regex[:-1] + ")+"
#new_abbr_regex = modify_abbr[:-1] + ")\\b"

# used in combination with the pure_abbr_reg
#pure_punct_reg = re.compile(("(?<=\n)(?=[%s]+)" % re.escape(punct)), re.IGNORECASE)

#new_abbr_reg = re.compile(new_abbr_regex)
# split punctuation apart.
# also want to be careful of abbreviations though
#punct = "'!\"#"
 #"(\d,? ?\d)|"
#"!\"#$%&'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`\{\|\}~"
abbr_regex = abbr_regex + "[%s]+" % re.escape(punct)
#global var reg
abbr_reg = re.compile(abbr_regex, re.IGNORECASE)


find_slash_reg = re.compile(r"\b\S+(?=/)") 


stopwords_f = open("/u/cs401/Wordlists/StopWords")
stopwords_set = set(stopwords_f.read().split())
stopwords_f.close()

#tokenized end of sentence detecting regex
end_of_sent_reg = re.compile("(?<=[!?.]/\S)")

extra_space_reg = re.compile(" {2,}")

def useRegExAbbrAndPunct(modComm):
    
    #capitalization still a potential problem
    reg_matches = abbr_reg.findall(modComm)
    #print("reg matches: ")
    #print(reg_matches)
    #for match in reg_matches:
        #match
    i = 0
    if reg_matches != []:
        for reg in reg_matches:
            if reg != "":
                length = len(reg)
                i = modComm.find(reg, i)
                end = i + length - 1
                #print(i, length, end)
                modComm = wordSpacer(i, end, modComm)
    return modComm
 

#my regexes lul
#"(https://|www\.|http://)(([a-zA-Z0-9])+\.[a-zA-Z0-9]+)" #this is just a website with no ////


#(https://|www\.|http://)(([a-zA-Z0-9])+\.[a-zA-Z0-9]+)(/[a-zA-Z0-9%s])*" % re.escape(string.punctuation))

def removeUrlFunction(sent):
    """removes all urls from a given string"""
    return url_reg.sub('', sent)



def wordSpacer(beg, end, sent):
    """want to split string 
    with indices sent[beg:end]
    and and add spaces around it"""
    # note! end = the last index of the last word

    new_str = ""

    if beg != 0 and sent[beg-1] != ' ':
        new_str = " " + sent[beg:end+1]
    else:
        new_str += sent[beg:end+1]
    if end != len(sent)-1 and sent[end+1] != ' ':
        new_str += ' '
    new_str = sent[:beg] + new_str + sent[end+1:]
    #new_str += sent[beg:end+1]
    #new_str += sent[end+1:]
    return new_str


def findPotentialAbbrev(period, sent):
    """Finds the potential word the period is part"""
    """period is the index of the period in sent,
    function returns potential word start index"""
    word_start = period
    while (not sent[word_start -1] in " \n") and (word_start-1 >= 0):
        word_start -= 1
    return word_start;


def preproc1(comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = comment;
    #print(modComm)
    if 1 in steps:

        modComm = modComm.replace("\n", ""); 
    if 2 in steps:

        modComm = html.unescape(modComm) 

    if 3 in steps:
        modComm = removeUrlFunction(modComm)
    #this step is removing urls. If url has a forward slash, accept everything until a space. Else, take only till the .(come/ca/gov, etc)   
        #print('TODO');
    if 4 in steps:


        modComm = useRegExAbbrAndPunct(modComm)
    if 5 in steps:
        modComm = clit_reg.sub(" ", modComm)

    if 6 in steps:
        modComm = extra_space_reg.sub(" ", modComm)
        utt = nlp(modComm)
        modComm = ""
        for token in utt:
            modComm += " " + token.text + "/" + token.tag_ 
        modComm = modComm[1:]

    if 7 in steps:

        modComm = stop_reg.sub("", modComm)
        modComm = extra_space_reg.sub(" ", modComm)

    if 8 in steps:
        #print('TODO');

        #modComm.replace("", ' $ ')
        comm_list = modComm.split()
        for i in range(len(comm_list)):
            token = find_slash_reg.findall(comm_list[i])
            if len(token) != 0:
                utt = nlp(token[0])
                if len(utt) != 0:
                    if utt[0].lemma_[0] != '-':
                        comm_list[i] = utt[0].lemma_ + '/' + utt[0].tag_
        modComm = " ".join(i for i in comm_list)

    if 9 in steps:

        modComm = end_of_sent_reg.sub("\n",modComm)
        
    if 10 in steps:
        #print('TODO');
        modComm = modComm.lower()

    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        file_counter = 1
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))
            print(os.walk(indir));
            print(args.ID);
            # TODO: select appropriate args.max lines
            maxIndex = len(data);
            index = (args.ID[0]%maxIndex);
            counter = 0; 
        # we have two seperate variables for index and counter, to make circular indexing easier
            # TODO: read those lines with something like `j = json.loads(line)`
        #j = json.load(data)
            while (counter < args.max):
                j = json.loads(data[index]);
                if (index ==  maxIndex-1):
                    index = 0;
                else:
                    index +=1;
                    counter += 1;
                j['body'] = preproc1(j['body'])         
                j['cat'] = file;
                allOutput.append(j);
        #print(line);
        #print(allOutput);      
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
        
    main(args)
    end = time.time()
    print(end - start)
