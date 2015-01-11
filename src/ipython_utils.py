from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from inspect import getmembers, isfunction, getsource
from IPython.core.display import HTML
import IPython

def print_python_file(p_fname):
	with open(p_fname) as f:
		code = f.read()

	formatter = HtmlFormatter()
	return HTML('<style type="text/css">{}</style>{}'.format(
	    formatter.get_style_defs('.highlight'),
	    highlight(code, PythonLexer(), formatter)))

def print_python_source(module, function):
    """For use inside an IPython notebook: given a module and a function, 
    print the source code."""

    internal_module = __import__(module)

    internal_functions = dict(getmembers(internal_module, isfunction))

    return HTML(highlight(getsource(internal_functions[function]), 
    			PythonLexer(), HtmlFormatter(full=True)))
