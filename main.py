from api import api
from config import Config

def run():
    '''
    Head of the program
    '''

    cnfg = Config()

    mode = cnfg.mode

    if mode is 'api':
        api(cnfg)
    else:
        raise Exception('Mode must be specified')
    

if __name__ == '__main__':
    run()