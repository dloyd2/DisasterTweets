import inspect, os
from datetime import datetime
from DisasterTweets.utility import LOCATION
log_storage = LOCATION+'/logs/'
if not os.path.exists(log_storage):
    os.mkdir(log_storage)
loggers = {}
def get_logger(name):
    '''
    name -> str

    Returns the log with the given name. If no such log exists, create it.    
    '''
    ret = None
    try:
        ret = loggers[name]
    except KeyError:
        ret = Logger(name, path=log_storage)
        loggers.update({name: ret})
    return ret

class Logger():
    def __init__(self, filename, path):
        self.filename = filename+'.log'
        self.path = path
        self.padding = '\t'
        caller_name = inspect.stack()[1][3]
        if caller_name.strip() != "get_logger":
            raise Exception('Use get_logger() to initalize a logger')
        f = open(self.path+self.filename, 'a')
        f.write("\n++++++++NEW LOG++++++++")
        f.close()
    def log(self, *args):
        '''
        Log each argument as a str, comma separated.
        Any input that can be turned into a str is valid
        '''
        new_entry = ", ".join([str(arg) for arg in args])
        f = open(self.path+self.filename, 'a')
        f.write("\n" + str(datetime.now()) + self.padding + new_entry)
        f.close()
