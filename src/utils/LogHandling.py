# TODO: Add code to detect the file and line or use a library for it
def log_err(message: str, disable: bool = False):
    if disable:
        log_disabled()
        return
    print('******\nERROR DETECTED: ' + str(message) + '\n******')


def log_prog(message: str, disable: bool = False):
    if disable:
        log_disabled()
        return
    print('CHECKPOINT:' + str(message))


def log_val(*args, value_name: str = '', disable: bool = False):
    if disable:
        log_disabled()
        return
    print('VALUE:', value_name, end = '')
    print(*args)


def log_disabled(disable: bool = False):
    if disable:
        return
    print('DISABLED')
