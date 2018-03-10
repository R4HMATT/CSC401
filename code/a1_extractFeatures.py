import numpy as np
import sys
import argparse
import os
import json
import re
import string
#def create_regex(filename)


#~~~Bunch of Regexes for Use in Steps 
with open("/u/cs401/Wordlists/First-person", 'r') as f:
    firstpron_list = f.read().split()
firstpron_regex = "("
for item in firstpron_list:
    firstpron_regex += r'\b'
    firstpron_regex += item
    # the slashes might be problematic"
    firstpron_regex += r'/|'
firstpron_regex = firstpron_regex[:-1] + ")"
firstpron_reg = re.compile(firstpron_regex)


secpron_regex = r"(?:^|\s)(you/|your/|yours/|u/|ur/|urs/)"
secpron_reg = re.compile(secpron_regex)

thirdpron_regex = r"(?:^|\s)(he/|him/|his/|she/|her/|hers/|it/|its/|they/|them/|their/|theirs/)"
thirdpron_reg = re.compile(thirdpron_regex)


coordpron_reg = re.compile(r"\w/CC ", re.IGNORECASE)

pasttense_reg = re.compile(r"\w/VBD|\w/VBN", re.IGNORECASE)

future_regex = r"(\w'll/| will/| gonna/|going token \w*/VB)"
future_reg = re.compile(future_regex, re.IGNORECASE)

commnouns_regex = r"(\S/NN(?=\b|\s)|\S/NNS(?=\b|\s))"
commnouns_reg = re.compile(commnouns_regex, re.IGNORECASE)

propnouns_regex = r"(\w/NNP(?=\b|\s)|\w/NNPS(?=\b|\s))"
propnouns_reg = re.compile(propnouns_regex, re.IGNORECASE)

adverb_regex = r"\w/RB(?=\b|\s)|\w/RBR(?=\b|\s)|\w/RBS(?=\b|\s)"
adverb_reg = re.compile(adverb_regex, re.IGNORECASE)

wh_regex = r"\w/WDT(?=\b|\s)|\w/WP(?=\b|\s)|\w/WP\$(?=\b|\s)|\w/WRB(?=\b|\s)"
wh_reg = re.compile(wh_regex, re.IGNORECASE)


with open("/u/cs401/Wordlists/Slang", 'r') as f:
    slang_list = f.read().split()
slang_regex = "(?:\b|\s)("
for item in slang_list:

    slang_regex += item
    # the slashes might be problematic"
    slang_regex += r'/|'
slang_regex = slang_regex[:-1] + ")"
slang_reg = re.compile(slang_regex, re.IGNORECASE)

def num_commas(comment):
    return comment.count(',')


#multi_punct_regex = r""""#\$%&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~]{2,}/"""
multi_punct_regex = '(?:^|\s)[%s]{2,}/\S' % re.escape(string.punctuation)
multi_punct_reg = re.compile(multi_punct_regex, re.IGNORECASE)

capitalword_reg = re.compile(r"(?:^|\s)[A-Z]{3,}/\w")

# find each token
token_reg = re.compile(r"(?:^|\s)[\S+]+(?=/\S)", re.IGNORECASE)



def average_length_of_sent(comment, num_sent):
    # assume each sentence is split with a \n after preprocessing
    total_tokens = 0
    sentences = comment.split('\n')
    for sent in sentences:
        total_tokens += len(token_reg.findall(sent))
    if num_sent == 0:
        val = 0
    else:
        val = total_tokens/num_sent
    return val

def num_sentences(comment):
    # assuming a \n char is between each sentence as per part 1 step
    return len(comment.split('\n'))


nonPunctOnly_regex = r"\b[a-zA-Z0-9_]+[%s]*/|\b[%s]*[a-zA-Z0-9_]+/|[a-zA-Z0-9_]+[%s]*[a-zA-Z0-9_]+/" % (string.punctuation, string.punctuation, string.punctuation)
nonPunctOnly_reg = re.compile(nonPunctOnly_regex, re.IGNORECASE)

AllPunctException_reg = re.compile("\w+/[A-Z]")
def avg_length_of_tokens(comment):
    token_list = nonPunctOnly_reg.findall(comment)
    total_letters = 0
    for item in token_list:
        total_letters += len(item)
    length = len(token_list)
    if length == 0:
        val = 0
    else:
        val = total_letters/length
    return val


BGL_norm_f = open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r')

BGL_data = BGL_norm_f.read().split('\n')[1:-3]

BGL_norm_f.close()
BGL_dict = {}
for line in BGL_data:
    curr = line.split(',')
    BGL_dict[curr[1]] = (curr[3], curr[4], curr[5])

war_dict = {}

war_file = open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv')
war_data = war_file.read().split('\n')[1:-1]
war_file.close()
for line in war_data:
    curr = line.split(',')
    war_dict[curr[1]] = (curr[2], curr[5], curr[8])

def stats(comment, return_array):
    token_list = token_reg.findall(comment)
    AoA = np.zeros([])
    img = np.zeros([])
    fam = np.zeros([])
    v = np.zeros([])
    a = np.zeros([]) 
    d = np.zeros([])
    for token in token_list:
        if token in BGL_dict:
            AoA = np.append(AoA, BGL_dict[token][0])
            img = np.append(img, BGL_dict[token][1])
            fam = np.append(fam, BGL_dict[token][2])
        if token in war_dict:
            v = np.append(v, war_dict[token][0])
            a = np.append(a, war_dict[token][1])
            d = np.append(d, war_dict[token][2])
    
    AoA = AoA.astype(np.float32)
    img = img.astype(np.float32)
    fam = fam.astype(np.float32)
    v = v.astype(np.float32)
    a = a.astype(np.float32)
    d = d.astype(np.float32)
    
    return_array = np.insert(return_array, 17, np.mean(AoA))
    return_array = np.insert(return_array, 18, np.mean(img))
    return_array = np.insert(return_array, 19, np.mean(fam))

    return_array = np.insert(return_array, 20, np.std(AoA))
    return_array = np.insert(return_array, 21, np.std(img))
    return_array = np.insert(return_array, 22, np.std(fam))

    return_array = np.insert(return_array, 23, np.mean(v))
    return_array = np.insert(return_array, 24, np.mean(a))
    return_array = np.insert(return_array, 25, np.mean(d))

    return_array = np.insert(return_array, 26, np.std(v))
    return_array = np.insert(return_array, 27, np.std(a))
    return_array = np.insert(return_array, 28, np.std(d))


    return return_array





# we already have a reg so we can get all the tokens
altID_dict = {}

altID_file = open('/u/cs401/A1/feats/Alt_IDs.txt', 'r')
alt_data = altID_file.read().split()
altID_file.close()
for i in range(len(alt_data)):
    altID_dict[alt_data[i]] = i


left_dict = {}

left_file = open('/u/cs401/A1/feats/Left_IDs.txt', 'r')
left_data = left_file.read().split()
left_file.close()
for i in range(len(left_data)):
    left_dict[left_data[i]] = i


right_dict = {}

right_file = open('/u/cs401/A1/feats/Right_IDs.txt', 'r')
right_data = right_file.read().split()
right_file.close()
for i in range(len(right_data)):
    right_dict[right_data[i]] = i

center_dict = {}

center_file = open('/u/cs401/A1/feats/Center_IDs.txt', 'r')
center_data = center_file.read().split()
center_file.close()
for i in range(len(center_data)):
    center_dict[center_data[i]] = i

altfeat_array = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
leftfeat_array = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
rightfeat_array = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
centerfeat_array = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
def extract1( comment ):
    ''' This function extracts features from a single comment
    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''
    array = np.zeros(144, np.float32)
    # TODO: your code here
    first_pronouns = len(firstpron_reg.findall(comment))
    sec_pron = len(secpron_reg.findall(comment))
    third_pron = len(thirdpron_reg.findall(comment))
    coord_pron = len(coordpron_reg.findall(comment))
    array = np.insert(array, 0, first_pronouns)
    array = np.insert(array, 1, sec_pron)
    array = np.insert(array, 2, third_pron)
    array = np.insert(array, 3, coord_pron)
    array = np.insert(array, 4, len(pasttense_reg.findall(comment)))
    #--coordinating conjunctions
    #--past tense
    array = np.insert(array, 5, len(future_reg.findall(comment)))
    array = np.insert(array, 6, comment.count(','))
    array = np.insert(array, 7, len(multi_punct_reg.findall(comment)))
    array = np.insert(array, 8, len(commnouns_reg.findall(comment)))

    array = np.insert(array, 9, len(propnouns_reg.findall(comment)))
    array = np.insert(array, 10, len(adverb_reg.findall(comment)))
    array = np.insert(array, 11, len(wh_reg.findall(comment)))
    array = np.insert(array, 12, len(slang_reg.findall(comment)))
    array = np.insert(array, 13, len(capitalword_reg.findall(comment)))

    num_sent = len(comment.split('\n'))

    array = np.insert(array, 14, average_length_of_sent(comment, num_sent))
    array = np.insert(array, 15, avg_length_of_tokens(comment))
    array = np.insert(array, 16, num_sent)


    return array



def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))
    # TODO: your code here

    for i in range(len(data)):
        curr_array = extract1(data[i]["body"])
        curr_array = stats(data[i]["body"], curr_array)
        it = np.nditer(curr_array, flags=['f_index'])
        cat = 0
        if data[i]['cat'] == 'Left':
            cat = 0
            index = left_dict[data[i]['id']]
            curr_array = np.concatenate((curr_array[:29], leftfeat_array[index]))
        elif data[i]['cat'] == 'Center':
            cat = 1
            index = center_dict[data[i]['id']]
            curr_array = np.concatenate((curr_array[:29], centerfeat_array[index]))
        elif data[i]['cat'] == 'Right':
            cat = 2
            index = right_dict[data[i]['id']]
            curr_array = np.concatenate((curr_array[:29], rightfeat_array[index]))
        elif data[i]['cat'] =='Alt':
            cat = 3
            index = altID_dict[data[i]['id']]
            curr_array = np.concatenate((curr_array[:29], altfeat_array[index]))
        curr_array = np.append(curr_array, cat)
        feats = np.insert(feats, i, curr_array, 0)
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

