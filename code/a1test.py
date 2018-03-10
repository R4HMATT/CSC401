 
import sys
import argparse
import os
import json


def main( args ):

    allOutput = [];
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file);
            print("Processing " + fullFile);

            data = json.load(open(fullFile));

            # TODO: select appropriate args.max lines
            max_lines = 10;
            # TODO: read those lines with something like `j = json.loads(line)`
	    j = json.loads
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)

         # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'

    print(data);

    fout = open(args.output, 'w');
    fout.write(json.dumps(allOutput));
    fout.close();

