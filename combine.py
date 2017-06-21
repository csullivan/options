#!/usr/bin/python3

import pandas, os, sys
import argparse


def get_program_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', type=str,help="Input binary options data files to combine.",default=None)
    parser.add_argument("--output", type=str,help="Output binary containing combined time series option data",default=None,required=True)
    args = parser.parse_args()
    return args;

if __name__ == "__main__":
    args = get_program_arguments()
    print("Number of files found: "+ str(len(args.input)))

    data = []
    for filename in args.input:
        chain = pandas.read_pickle(filename)
        try:
            chain.set_index('Quote_Time', append=True, inplace=True)
            chain = chain.swaplevel('Strike','Expiry').sort_index(level=0)
            chain = chain.drop('JSON',1)
            chain.reset_index(level='Symbol')
        except:
            pass
        data.append(chain)

    combined = pandas.concat(data,axis=0).sort_index(level=0)
    combined = combined.drop_duplicates().sort_index(level=0)
    #combined = combined.reset_index(level='Symbol')
    combined.to_pickle(args.output)
