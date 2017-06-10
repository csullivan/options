#!/usr/bin/python3

import time
import datetime

from twisted.internet import task
from twisted.internet import reactor
from pandas_datareader.data import Options
from pandas import HDFStore
from time import gmtime, strftime


def hdf_store(df,filename):
    hdf = HDFStore(filename+'.h5')
    hdf.put('spy', df, format='table', data_columns=True)
    hdf.close()

def pickle_store(df,filename):
    df.to_pickle(filename)

def get_data():
    spy = Options('spy', 'yahoo')
    chain = spy.get_all_data()
    filename = './spy_'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'.bin'
    pickle_store(chain,filename)
    #chain = chain.drop('JSON',1)
    #hdf_store(chain,filename)
    #exit_at(datetime.datetime(2017,6,8,16,2)) # 4:02pm

def delay_until(schedule_time):
    while True:
        dateSTR = datetime.datetime.now().strftime("%H:%M:%S" )
        if dateSTR == (scheduled_time):
            print(dateSTR)
            break
        else:
            time.sleep(1)

def exit_at(schedule_time):
    if (schedule_time-datetime.datetime.now().total_seconds() < 0):
        print("Done!")
        exit(0);
    else:
        print("Query finished. Sleeping..")
        return

if __name__ == "__main__":
    scheduled_time = '09:25:00'
    delay_until(scheduled_time)

    #get_data()
    timeout = 60.0
    l = task.LoopingCall(get_data)
    l.start(timeout)
    reactor.run()
