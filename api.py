from config import Config

def api(cnfg: Config):
    '''
    Run if the mode is API
    '''

    cnfg.use_stream = False
    