from subprocess import call, check_call
import csv
import sys
from time import sleep

# This should be copied to then run from an MSL server
# Then you copy the created directory to your local machine

data_dir = '/molokini_raid/MSL/data/surface/processed/images/web/full/SURFACE/'
def get_raw_path(instrument, sol, name):
    return data_dir + '%s/%s/%s' % (
        instrument,
        'sol' + str(sol).zfill(4),
        name.strip('\"') + '.png'
    )

csv_fn = sys.argv[1]
dir_name = csv_fn[:-4]
call(['mkdir', './' + dir_name])
with open(csv_fn, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        name, instrument, sol, filter_used, width, height = row
        call(['cp',
            get_raw_path(instrument, sol, name),
            dir_name])