import string
import sys
import argparse
import os
import re
import json
#from html import * 
indir = '/u/cs401/A1/data/';


#my regexes lul
#"(https://|www\.|http://)(([a-zA-Z0-9])+\.[a-zA-Z0-9]+)" #this is just a website with no ////


#(https://|www\.|http://)(([a-zA-Z0-9])+\.[a-zA-Z0-9]+)(/[a-zA-Z0-9%s])*" % re.escape(string.punctuation))

def removeUrlFunction(sent):
    """removes all urls from a given string"""
    reg = re.compile("(https://|www\.|http://)(([a-zA-Z0-9])+\.[a-zA-Z0-9]+)(/[a-zA-Z0-9%s])*" % re.escape(string.punctuation))
    return reg.sub('', sent)



def wordSpacer(beg, end, sent):
    """want to split string 
    with indices sent[beg:end]
    and and add spaces around it"""
    # note! end = the last index of the last word

    new_str = sent[:beg]

    if sent[beg-1] not in  " \n" and beg != 0:
        new_str += " "
    new_str += sent[beg:end+1]
    if sent[end] not in " \n" and end != len(sent)-1 :
        new_str += " "
    new_str += sent[end+1:]
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
    if 1 in steps:
        print('TODO');
        modComm = comment.replace("\n", ""); 
    if 2 in steps:
        print('TODO');
    # check to see if this method worked or not, on the html (it works to convert in unicode, but not exactly ascii as assignment wants)
    #modComm = html.unescape(modComm);
    if 3 in steps:
    #this step is removing urls. If url has a forward slash, accept everything until a space. Else, take only till the .(come/ca/gov, etc)   
        print('TODO');
    if 4 in steps:
        # open file of abbreviations
        f = open(r"/u/cs401/Wordlists/abbrev.english");
        print("file opened, so i guess thats not the problem")
        abbr = f.read()
        f.close();
        abbr = abbr.strip()
        abbr = abbr.split('\n')
        print('TODO');
        # split punctuation apart.
        # also want to be careful of abbreviations though
        punct = string.punctuation.replace("'", "");
        punct = punct.replace('-', "");
        i = 0;

        while(i != len(modComm)):
            #if curr index is a punctuation mark
            if modComm[i] in punct:
                # find space before the punctuation mark, to find the word it is (potentially)
                space = modComm.rfind(" ", 0, i);
                word = space + 1; # word is where the potential word begins
                # if there wasnt a space behind the punctuation
                if space == -1:
                    word  = i;
                #if the word is in the abbr, do nothing
                if  modComm[word:i+1] in abbr:
                    pass
                # if its not
                else:
                    # check if there's following punctuation
                    after_punct = i+1; # this is the index of the end of a string of punctuations, by default the string of punctuation is one punctuation mark long

                    while modComm[after_punct] in punct:
                        after_punct+= 1
                    modComm = modComm[:i] + "" + modComm[i:after_punct] + "" + modComm[after_punct:]
            i+=1
    if 5 in steps:
        print('TODO');
    if 6 in steps:
        print('TODO');
    if 7 in steps:
        print('TODO');
    if 8 in steps:
        print('TODO');
    if 9 in steps:
        print('TODO');
    if 10 in steps:
        print('TODO');
        
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print "Processing " + fullFile

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

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print "Error: If you want to read more than 200,272 comments per file, you have to read them all."
        sys.exit(1)
        
    main(args)
