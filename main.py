from api import api
from config import Config
from console import console
from webui import webui

def run():
    '''
    Head of the program
    '''

    cnfg = Config()

    mode = cnfg.mode

    if mode is 'api':
        api(cnfg)
    elif mode == 'api':
        api(cnfg)
    elif mode == 'webui':
        webui(cnfg)
    else:
        raise Exception('Mode must be specified')
    

if __name__ == '__main__':
    run()

