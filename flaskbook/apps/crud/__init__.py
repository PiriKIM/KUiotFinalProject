from flask import Blueprint

sample = Blueprint(
    'sample',
    __name__,
    static_folder='static',
    template_folder='templates',
    url_prefix='/sample',
    subdomain='example'
)

@sample.route('/')
def index():
    return "Hello, Sample!"