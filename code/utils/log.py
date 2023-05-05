import os
import shutil
from constants.CONFIG import *
from datetime import datetime
import pytz


def make_log_dirs(dir_path):
    """my_log 안에 날짜폴더를 생성, config.json파일을 복사"""
    kr_tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz=kr_tz)
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S')
    folder_path = os.path.join(dir_path, folder_name)

    os.mkdir(folder_path)
    shutil.copyfile(CONFIG_PATH, os.path.join(folder_path, CONFIG_NAME))

    return folder_path
