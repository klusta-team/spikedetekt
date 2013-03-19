'''
Log and warning functions.

Call log_message(msg) for a normal message, or log_warning(msg) to indicate a
potential error. The number of warnings will be reported at the end of the
log file.

For messages with multiple lines, use the keyword multiline=True, it assumes
that your message looks like:

    """
    Here is a message.
    With
    multiple
    lines.
    """
    
And it will print it with the whitespace at the beginning of each line stripped
out.
'''
from parameters import GlobalVariables
__all__ = ['log_message',
           'log_warning']

def log_message(msg, multiline=False):
    log_fd = GlobalVariables['log_fd']
    if multiline:
        msg = '\n'.join([l.strip() for l in msg.split('\n')[1:]])
    print msg
    if log_fd is not None:
        log_fd.write(msg+'\n')
        
def log_warning(msg, multiline=False):
    GlobalVariables['warnings'] += 1
    log_message(msg, multiline)
