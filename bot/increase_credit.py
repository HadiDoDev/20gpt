# import os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
# print(cwd)
#os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")

import argparse
import database

parser = argparse.ArgumentParser(description='Script so useful to increase user credit.')
parser.add_argument("--user_id", type=int)
parser.add_argument("--total_rials", type=int)
parser.add_argument("--chat_modes", "--list", nargs='+')

args = parser.parse_args()

db = database.Database()


if __name__ == "__main__":
    db.increase_user_credit(args.user_id, args.total_rials, args.chat_modes)
