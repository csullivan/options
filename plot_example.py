#!/usr/bin/python3
import seaborn
import matplotlib.pyplot as plt
import pandas
seaborn.set_context('paper',font_scale=2.5)

if __name__ == "__main__":
    strike = 240.
    combined = pandas.read_pickle('spy_2017-06-07_full2.bin')
    print ('Strike: '+str(strike))
    ax = combined.loc['2017-06-09',strike,'call'].plot(y='Underlying_Price',use_index=True)
    ax.set_xlim(pandas.Timestamp('2017-06-07 13:30:00'),pandas.Timestamp('2017-06-07 20:30:00'))
    plt.show()

