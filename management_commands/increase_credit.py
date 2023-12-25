import argparse
from ..bot import database
from ..bot import config

parser = argparse.ArgumentParser(description='Script so useful to increase user credit.')
parser.add_argument("--user_id", type=int)
parser.add_argument("--total_rials", type=int)
parser.add_argument("--chat_modes", type=list)

args = parser.parse_args()

db = database.Database(config.mongodb_uri)


if __name__ == "__main__":
    db.increase_user_credit(args.user_id, args.total_rials, args.chat_modes)
