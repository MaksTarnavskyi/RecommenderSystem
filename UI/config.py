import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    current_path = os.getcwd()

    WORK_FOLDER = current_path + '/app/data/'
    PATH_SAVE = current_path + '/app/saved_files/'
